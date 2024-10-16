#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

//use anyhow::Ok;
use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};

use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;

use candle_transformers::models::mamba::{Config, Model, State};
use std::cell::Ref;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::ops::Add;

use candle::Result as CResult;
use candle::{DType, Device, Tensor, Var};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::{VarBuilder, VarMap};
use candle_optimisers::{self, lbfgs, LossOptimizer, ModelOutcome, Trainable};
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Clone)]
struct Trainer {
    pub model: Model,
    pub vars: VarMap,
    data: Vec<String>,
    config: Config,
    device: Device,
    dtype: DType,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl Trainable for Trainer {
    //TODO: make this leaner but first ensure recurrent mode state is correct for loss calc
    fn loss(&mut self) -> CResult<Tensor> {
        // println!("data: {}", self.data.last().unwrap());
        self.reset_state().unwrap();
        use std::io::Write;
        self.tokenizer.clear();
        let dtype = self.model.dtype();

        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(self.data.last().unwrap().clone(), true)
            .map_err(E::msg)
            .unwrap()
            .get_ids()
            .to_vec();
        let orig_token_len = tokens.len() as f32;
        let orig_token_len = Tensor::new(&[orig_token_len], &self.device)?.squeeze(0)?;
        //        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
        //            Some(token) => token,
        //            //None => anyhow::bail!("cannot find the </s> token"),
        //            None => panic!("cannot find the /s token!"),
        //        };
        let mut next_logits = None;
        let label_token = tokens.last().unwrap();
        //TODO: this should be determined in the dataset chunk function
        let mut pred_tokens = 0;
        println!("tokens len: {}", tokens.len());
        if tokens.len() > 2 {
            // pred_tokens = tokens.len()/4;
            pred_tokens = tokens.len() - 1 as usize; // at least conv_d
                                                     //            pred_tokens = tokens.len() - (  tokens.len() as f32 *0.80) as usize; // at least conv_d
        } else {
            pred_tokens = 1;
        }
        let token_labels = tokens.clone();
        let label_tokens = token_labels
            .iter()
            .rev()
            .take(pred_tokens)
            .collect::<Vec<&u32>>();

        //TODO: if we do it like this without parallel conv, we need to backprop every token otherwise this is likea liquid time-series network
        for &t in tokens.iter().take(tokens.len() - pred_tokens) {
            let input = Tensor::new(&[t], &self.device)?;
            self.model.input = Some(input.clone());
            let logits = self.model.forward_state()?;

            next_logits = Some(logits);
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("|{t}|")
            }
            //TODO: save state, predict the next token, calculate accumulate loss then reset state
        }

        std::io::stdout().flush()?;
        //:wtokens.push(next_token);

        //Calculate label logits and CrossEntropy
        //let mut losses = Vec::new();
        self.tokenizer.clear();
        let mut loss = Tensor::new(&[0.0], &self.device)?
            .to_dtype(dtype)?
            .squeeze(0)?;
        let mut iter = 0 as f32;
        for label_token in label_tokens.into_iter().rev() {
            iter += 1.;
            let itert = Tensor::new(&[iter], &self.device)?.squeeze(0);
            let logits = match next_logits.as_ref() {
                Some(logits) => logits,
                None => panic!("cannot train on an empty prompt"),
            };
            let pred_logits = logits.clone().to_dtype(dtype)?;

            //:let next_token = self.logits_processor.sample(&logits)?;
            let next_token = self
                .logits_processor
                .sample(&pred_logits.squeeze(0)?.to_dtype(dtype)?)?;
            // if next_token == eos_token {
            //     break;
            // }
            let next_prediction_token = self.tokenizer.next_token(next_token)?;
            //            if let Some(t) = self.tokenizer.next_token(next_token)? {
            if let Some(t) = next_prediction_token.clone() {
                print!("[{t}]");
                //            std::io::stdout().flush()?;
            } else {
                print!("[-]");
            }
            //            println!("no pred");
            let label = Tensor::new(&[label_token.clone()], &self.device)?;
            // let label = Tensor::new(&[label_token.clone()], &self.device)?.to_dtype(dtype)?;
            // println!("label: {}", label);
            // println!("vs: {}", next_token);
            //TODO: ensure label is the logit position

            //self.model.label = Some(label);

            //TODO: mini batch this, accumulate labels and tokens and pass all of them to cross_entropy so we get proper logits not binary masking this can be done with a iter.map
            //            let next_token = Tensor::new(&[next_token.clone()], &self.device)?.to_dtype(dtype)?;
            loss = (loss
                + itert
                    * candle_nn::loss::cross_entropy(&pred_logits, &label)?
                        .div(&orig_token_len.clone())?)?;
            //            loss = ( loss
            //                + candle_nn::loss::cross_entropy(&pred_logits.clone(), &label.clone())?)?;
            //losses.push(loss.clone().to_scalar::<f32>()?);
            //let loss = candle_nn::loss::mse(&label, &next_token);
            //let other_loss = candle_nn::loss::mse(&label, &next_token)?;
            //            let other_loss =
            //                candle_nn::loss::cross_entropy(&pred_logits.clone(), &label.clone()).unwrap();
            //         println!("{other_loss}");

            //                        let input = Tensor::new(&[label_token.clone()], &self.device)?;
            let input = Tensor::new(&[next_token.clone()], &self.device)?;
            //            self.model.input = Some(input.clone());
            self.model.input = Some(input.clone());
            let logits = self.model.forward_state()?;

            next_logits = Some(logits);
        }
        //let mut f_new = f_new.to_dtype(candle::DType::F64)?.to_scalar::<f64>()?;
        tokens.clear();
        self.tokenizer.clear();
        if loss.as_ref().clone().to_scalar::<f32>()?.is_nan() {
            println!("nan");
            //TODO: this should be extracted to the optimizer
            //TODO: figure out what max can be for cross_entropy given vector size and logits and set this value more reasonably (minimize)
            loss = Tensor::new(&[f32::MAX], &self.device)?
                //            loss = Tensor::new(&[10000000.0], &self.device)?
                .to_dtype(dtype)
                .unwrap()
                .squeeze(0)?;
        }
        return Ok(loss);
        //        return Ok(Tensor::new(losses,&self.device)?.to_dtype(dtype)?);
    }
}
impl Trainer {
    #[allow(clippy::too_many_arguments)]
    fn new() -> Result<Self> {
        use std::str::FromStr;
        use tracing_chrome::ChromeLayerBuilder;
        use tracing_subscriber::prelude::*;

        let args = Args::parse();
        let seed = args.seed;
        let temp = args.temperature;
        let top_p = args.top_p;
        let repeat_penalty = args.repeat_penalty;
        let repeat_last_n = args.repeat_last_n;
        let _guard = if args.tracing {
            let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
            tracing_subscriber::registry().with(chrome_layer).init();
            Some(guard)
        } else {
            None
        };
        println!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle::utils::with_avx(),
            candle::utils::with_neon(),
            candle::utils::with_simd128(),
            candle::utils::with_f16c()
        );
        println!(
            "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
            args.temperature.unwrap_or(0.),
            args.repeat_penalty,
            args.repeat_last_n
        );

        let start = std::time::Instant::now();
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            args.model_id
                .unwrap_or_else(|| args.which.model_id().to_string()),
            RepoType::Model,
            args.revision
                .unwrap_or_else(|| args.which.revision().to_string()),
        ));
        let tokenizer_filename = match args.tokenizer_file {
            Some(file) => std::path::PathBuf::from(file),
            None => api
                .model("EleutherAI/gpt-neox-20b".to_string())
                .get("tokenizer.json")?,
        };
        let config_filename = match args.config_file {
            Some(file) => std::path::PathBuf::from(file),
            None => repo.get("config.json")?,
        };
        let filenames = match args.weight_files {
            Some(files) => files
                .split(',')
                .map(std::path::PathBuf::from)
                .collect::<Vec<_>>(),
            None => {
                vec![repo.get("model.safetensors")?]
            }
        };
        println!("retrieved the files in {:?}", start.elapsed());
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let start = std::time::Instant::now();
        //load hyperparameters for this model as served from HuggingFace repo
        let read_config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let config: Config = Config {
            d_model: 40,
            n_layer: 40,
            vocab_size: read_config.vocab_size,
            pad_vocab_size_multiple: read_config.pad_vocab_size_multiple,
        };
        //pub struct Config {
        //    pub d_model: usize,
        //    pub n_layer: usize,
        //    pub vocab_size: usize,
        //    pub pad_vocab_size_multiple: usize,
        //}

        let device = candle_examples::device(args.cpu)?;
        let dtype = DType::from_str(&args.dtype)?;
        //load the weight tensors for training
        //let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        let mut vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, dtype, &device);
        //TODO: extract to initialization routine we need this alot in train
        let state = State::new(1, &config, dtype, &device)?;
        let mut model = Model::new(&config, vb.pp("backbone"), state)?;
        //vm.load_multi(&filenames)?;
        println!("loaded the model in {:?}", start.elapsed());
        println!("loaded: {} trainable Tensors", vm.all_vars().len());

        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Ok(Self {
            model,
            vars: vm,
            data: vec![],
            config,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
            dtype: dtype,
        })
    }

    pub fn add_data(&mut self, prompt: &str) {
        self.data.push(prompt.to_string());
    }

    pub fn add_dataset(&mut self, prompts: Vec<String>) {
        for data in prompts {
            if data.len() > 10 {
                self.add_data(data.as_str());
            }
        }
    }
    pub fn reset_state(&mut self) -> Result<()> {
        let state = State::new(1, &self.config, self.dtype, &self.device)?;
        self.model.state.hs.clear();
        self.model.state.prev_xs.clear();
        self.model.state = state;
        Ok(())
    }

    //TODO: clean this up, almost all of this can be refactored into the lbfgs optimiser
    fn train(
        &mut self,
        lbfgs_state: Option<lbfgs::lbfgs_state>,
        iter: usize,
        mut converged: bool,
    ) -> Result<(Tensor, Option<lbfgs::lbfgs_state>)> {
        let mut vars = self.vars.all_vars().clone();
        let mut loss = self.loss().unwrap().clone();
        //let new_loss = loss.clone();
        println!("begin training..");
        //TODO: LBFGS leaks cuda data structures slowly.

        let params = lbfgs::ParamsLBFGS {
            //            lr: 0.1,
            history_size: 20,
            line_search: Some(lbfgs::LineSearch::StrongWolfe(1e-4, 0.9, 1e-13)),
            //            weight_decay: Some(1e-9), // NOTE: this must be critically dampened with the learning rate
            ..Default::default()
        };
        let selfi = &mut self.clone();
        let mut lbfgs_opt = lbfgs::Lbfgs::new(vars, params, selfi)?;
        if lbfgs_state.is_some() {
            println!("loading Hessian..");
            lbfgs_opt.load_state(lbfgs_state.unwrap());
            //            lbfgs_opt.next_grad = None;
        }
        let mut reset = false;
        let mut hessian_pop = false;
        let mut vanished = false; //TODO: this should be solved in the LBFGS optimizer implementation and is currently a workaround.
        let mut prev_loss = f32::MAX;
        let mut prev_loss = Tensor::new(&[prev_loss], &self.device)?;
        //        for i in 0..3*lbfgs_opt.params.history_size {
        for i in 0..25 {
            //        loop {
            //                    for i in 0..5{
            let res = lbfgs_opt.backward_step(&mut loss).unwrap();
            match res {
                ModelOutcome::Converged(new_loss, _) => {
                    loss = new_loss.clone();
                    //                    if i < 1 {
                    //                        loss = Tensor::new(&[f32::MAX], &self.clone().device.clone())?
                    //                            .squeeze(0)?;
                    hessian_pop = true;
                    vanished = true;
                    reset = true;
                    break;
                    //                    //                    }
                    //                    break;
                }
                ModelOutcome::GradConverged(new_loss, _) => {
                    loss = new_loss.clone();
                    reset = true;
                    //                    //                    if i < 1 {
                    hessian_pop = true;
                    vanished = true;
                    println!("GRAD CONVERGED");
                    //                    //                    }
                    break;
                    //loss
                }
                ModelOutcome::Stepped(new_loss, _) => {
                    //loss
                    prev_loss = loss.clone();
                    print!(
                        "\t \t prev loss: {}  ->  ",
                        loss.clone().to_scalar::<f32>()?
                    );
                    loss = new_loss.clone();
                    println!("\t \t cur loss: {}", loss.clone().to_scalar::<f32>()?);
                    //                    if converged{
                    //                    lbfgs_opt.first = true;
                    //                    lbfgs_opt.s_hist.clear();
                    //                    lbfgs_opt.next_grad=None;
                    //                                        lbfgs_opt.last_grad = None;
                    //                                        lbfgs_opt.last_step = None;
                    //}
                    //                    if lbfgs_opt.params.lr > 0.0001 {
                    //                        lbfgs_opt.params.lr -= 0.0001;
                    //                    }
                }

                _ => panic!("unexpected outcome"),
            }
            //Handle if the gradient vanishes or explodes
            if (loss.clone().to_scalar::<f32>()? <= 0.0)
            //                || loss.clone().to_scalar::<f32>()? >= prev_loss.clone().to_scalar::<f32>()?)
            //                || loss.clone().to_scalar::<f32>()?.is_nan())
            {
                lbfgs_opt.next_grad = None;
                lbfgs_opt.last_grad = None;
                lbfgs_opt.last_step = None;
                break;
            }
            //            if (loss.clone().to_scalar::<f32>()? == f32::MAX)
        }
        println!("\t \t \t final loss: {}", loss.clone());
        //        //TODO: we can keep the hessian if we stop early when loss isnt increasing
        //                lbfgs_opt.s_hist.clear();
        //        lbfgs_opt.first = true;
        lbfgs_opt.next_grad = None;
        println!("s_hist: {}", lbfgs_opt.s_hist.len());
        converged = false;
        //                lbfgs_opt.last_grad = None;
        //                lbfgs_opt.last_step = None;
        //                //selfi.model.layers.clear();
        //TODO: move this into each match above and dont break the loop unless error is 0
        if reset {
            println!("RESET");
            if vanished {
                if hessian_pop {
                    //                    lbfgs_state.lr += 0.0001;
                    let mut rng = rand::thread_rng();
                    //    let num_entries = rng.gen_range(1..=iterable.len()); // Generate a random number between 1 and the length of the iterable

                    // Shuffle the iterable and take the first `num_entries`
                    //TODO: extract a hyperparameter and ensure we randomly select for all_vars
                    let mut shuffled_iterable = self.vars.all_vars();
                    let num_entries = (shuffled_iterable.len() as f32 * 0.1) as usize;
                    shuffled_iterable.shuffle(&mut rng);

                    shuffled_iterable
                        .into_iter()
                        .take(num_entries)
                        .for_each(|x| {
                            //let mut sparse = &x.clone().as_tensor().broadcast_mul(&Tensor::new(0. as f32, &self.device).unwrap()).unwrap();
                            let mut vec_tens = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                            for i in 0..vec_tens.len() * 0.1 as usize {
                                let first_idx = rng.gen_range(0..=vec_tens.len() - 1); // Generate a random number between 1 and the length of the iterator
                                                                                       //    let second_idx = rng.gen_range(1..=vec_tens[first_idx].len()); // Generate a random number between 1 and the length of the iterator
                                                                                       //vec_tens[0][first_idx][second_idx] =              0. as f32;
                                                                                       //TODO: this would work much more efficiently with a basic momentum. prevents catastrophic forgetting and convergence loop
                                let random_float: f32 = rng.gen_range(-1.0..=1.0);
                                //    let random_float: f32 = 1. as f32;
                                //    let random_float: f32 = rng.gen();
                                vec_tens[first_idx] = random_float;
                                //                                                                                       vec_tens[first_idx] =              rng.gen_range::<f32,f32>(-1 as f32..1 as f32) as f32;
                                //                                vec_tens[first_idx] = 1. as f32;

                                //'kor::new(&[0. as f32], &self.device)?;
                            }
                            let sparse =
                                &Tensor::from_vec(vec_tens, x.shape(), &self.device).unwrap();
                            //                        let sparse = &x
                            //                            .as_tensor();
                            //                            .broadcast_add(&Tensor::new(100 as f32, &self.device).unwrap())
                            //                            .unwrap();
                            //                        //                            sparse.flatten_all().into_iter().for_each(|mut x| {
                            //                                x = Tensor::new(0., &self.device).unwrap();
                            //                            });
                            x.set(sparse);
                            println!("shape: {:?}", x.as_tensor());
                            //println!("got: {}", x);
                        });
                    //CTMFIT
                    //                                        lbfgs_opt.s_hist.clear();
                    //                                        lbfgs_opt.last_grad = None;
                    //                                        lbfgs_opt.last_step = None;
                    lbfgs_opt.next_grad = None;
                    lbfgs_opt.first = true;
                    //                    println!("-------------------------------------CLEARING Stuck Hessian {}..----------------------------------------------------", lbfgs_state.s_hist.len());
                    println!("-------------------------------------CLEARING Stuck Hessian {}..----------------------------------------------------", 0.);
                }
                //                lbfgs_state.s_hist.pop_back();
            }
        }
        let mut lbfgs_state = lbfgs_opt.save_state();
        //        lbfgs_state.s_hist.clear();
        //        lbfgs_state.last_grad = None;
        //        lbfgs_state.last_step = None;
        //        lbfgs_state.next_grad = None;
        //        lbfgs_state.first = true;
        Ok((loss.copy()?, Some(lbfgs_state)))
    }

    fn run_trained(&mut self) -> Result<()> {
        // println!("data: {}", self.data.last().unwrap());
        self.reset_state().unwrap();
        use std::io::Write;
        self.tokenizer.clear();
        let dtype = self.model.dtype();
        println!("data: {}", self.data.last().unwrap());

        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(self.data.last().unwrap().clone(), true)
            .map_err(E::msg)
            .unwrap()
            .get_ids()
            .to_vec();

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the </s> token"),
        };
        let mut next_logits = None;
        for &t in tokens.iter().take(tokens.len() - 1) {
            let input = Tensor::new(&[t], &self.device)?;
            self.model.input = Some(input);
            let logits = self.model.forward_state()?;

            next_logits = Some(logits);
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let start_gen = std::time::Instant::now();
        // for _ in 0..sample_len {
        let logits = match next_logits.as_ref() {
            Some(logits) => logits,
            None => anyhow::bail!("cannot work on an empty prompt"),
        };
        let logits = logits.squeeze(0)?.to_dtype(dtype)?;
        // let logits = if self.repeat_penalty == 1. {
        //     logits
        // } else {
        //     let start_at = tokens.len().saturating_sub(self.repeat_last_n);
        //     candle_transformers::utils::apply_repeat_penalty(
        //         &logits,
        //         self.repeat_penalty,
        //         &tokens[start_at..],
        //     )?
        // };
        let next_token = self.logits_processor.sample(&logits)?;
        tokens.push(next_token);
        generated_tokens += 1;
        // if next_token == eos_token {
        //     break;
        // }
        if let Some(t) = self.tokenizer.next_token(next_token)? {
            print!("\t {t}");
            std::io::stdout().flush()?;
        }

        if let Some(t) = self.tokenizer.next_token(next_token)? {
            print!("\t {t}");
            std::io::stdout().flush()?;
        }
        let input = Tensor::new(&[next_token], &self.device)?;
        self.model.input = Some(input);
        next_logits = Some(self.model.forward_state()?);
        //next_logits = Some(self.model.forward(&input, &mut self.model.state)?)
        // }
        let dt = start_gen.elapsed();

        std::io::stdout().flush()?;
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("\t\t rest: {rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.reset_state();
        self.tokenizer.clear();
        let dtype = self.model.dtype();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the </s> token"),
        };
        let mut next_logits = None;
        for &t in tokens.iter() {
            let input = Tensor::new(&[t], &self.device)?;
            self.model.input = Some(input);
            let logits = self.model.forward_state()?;

            next_logits = Some(logits);
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let start_gen = std::time::Instant::now();
        for _ in 0..sample_len {
            let logits = match next_logits.as_ref() {
                Some(logits) => logits,
                None => anyhow::bail!("cannot work on an empty prompt"),
            };
            let logits = logits.squeeze(0)?.to_dtype(dtype)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("\t {t}");
                std::io::stdout().flush()?;
            }

            let input = Tensor::new(&[next_token], &self.device)?;
            self.model.input = Some(input);
            next_logits = Some(self.model.forward_state()?)
            //next_logits = Some(self.model.forward(&input, &mut self.model.state)?)
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Parser, ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
enum Which {
    Mamba130m,
    Mamba370m,
    Mamba790m,
    Mamba1_4b,
    Mamba2_8b,
    Mamba2_8bSlimPj,
}

impl std::fmt::Display for Which {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Which {
    fn model_id(&self) -> &'static str {
        match self {
            Self::Mamba130m => "state-spaces/mamba-130m",
            Self::Mamba370m => "state-spaces/mamba-370m",
            Self::Mamba790m => "state-spaces/mamba-790m",
            Self::Mamba1_4b => "state-spaces/mamba-1.4b",
            Self::Mamba2_8b => "state-spaces/mamba-2.8b",
            Self::Mamba2_8bSlimPj => "state-spaces/mamba-2.8b-slimpj'",
        }
    }

    fn revision(&self) -> &'static str {
        match self {
            Self::Mamba130m
            | Self::Mamba370m
            | Self::Mamba790m
            | Self::Mamba1_4b
            | Self::Mamba2_8bSlimPj => "refs/pr/1",
            Self::Mamba2_8b => "refs/pr/4",
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 5000)]
    sample_len: usize,

    #[arg(long, default_value = "mamba130m")]
    which: Which,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long, default_value = "f32")]
    dtype: String,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

use std::fs::File;
use std::io::{self, prelude::*, BufReader};
use std::path::Path;

fn lines_from_file(filename: impl AsRef<Path>) -> Vec<String> {
    let mut file = File::open(filename).expect("no such file");
    let mut contents = String::new();
    file.read_to_string(&mut contents);
    contents
        .as_bytes()
        //TODO: this should be by tokens so we dont get memory allocation variance on different samples
        .chunks(500)
        .map(|chunk| String::from_utf8_lossy(chunk).into_owned())
        .collect()
    //    let buf = BufReader::new(file);
    //    buf.lines()
    //        .filter(|e| e.as_ref().clone().unwrap().len() > 0)
    //        .map(|l| l.expect("Could not parse line"))
    //        .collect()
}

fn main() -> Result<()> {
    //TODO: replace this with a class for training
    //let prompts = vec!["mamba is a model", "mamba, the model", "mamba is the model"];
    //let prompts = lines_from_file("README.md");
    // for i in 0..10 * learner.data.len() {
    let mut prompts = lines_from_file("TEXT.txt");
    let mut opt_state = None;
    // for _ in 0..2 * learner.data.len() {
    let mut i = 1;
    let mut converged = false;
    let mut learner = Trainer::new()?;
    let mut model = learner.model.clone();
    let mut varmap = learner.vars.clone();
    for mut j in 1..11 * prompts.len() {
        let mut learner = Trainer::new()?;
        learner.add_dataset(prompts.clone());
        learner.model = model.clone();
        learner.vars = varmap.clone();
        learner.data.rotate_left(j);
        opt_state = learner.train(opt_state.clone(), 10, converged)?.1;
        i = j;
        //learner.run("mamba is a", 20);
        println!("done training!");
        //learner.data.rotate_right(1);
        //let err = learner.run_trained().unwrap();
        //learner.data.rotate_left(1);
        learner.run("Rust is  ", 20);
    }
    //learner.run("mamba is a", 20);
    //model,
    //config,
    //tokenizer,
    //args.seed,
    //args.temperature,
    //args.top_p,
    //args.repeat_penalty,
    //args.repeat_last_n,
    //&device,
    //);
    //pipeline.run(&args.prompt, args.sample_len)?;

    //let loss = pipeline.train(&args.prompt, vm.all_vars())?;

    //TODO: extract the pipeline so we get the same state iteration per backward_step
    //TODO: train needs to include lbfgs, reset the state of the model each time and iterate
    //    for _ in 0..10 {
    //        let res = lbfgs_opt.backward_step(&loss)?;
    //        match res {
    //            ModelOutcome::Converged(new_loss, _) => {
    //                println!("converged {}", new_loss);
    //                let loss = new_loss;
    //            }
    //            ModelOutcome::Stepped(new_loss, _) => {
    //                let loss = new_loss;
    //                println!("loss {}", loss);
    //            }
    //        }
    //model.state = State::new(1, &config, dtype, &device)?;//TODO: need to do this in pipeline.train each time
    //}
    Ok(())
}
