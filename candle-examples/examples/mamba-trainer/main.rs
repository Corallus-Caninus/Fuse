#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};

use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;

//use candle_transformers::models::mamba::{Config, Model, State};
mod model;
use model::{Config, Model};
use std::cell::Ref;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::ops::Add;

use candle::Result as CResult;
use candle::{DType, Device, Module, Tensor, Var};
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
    fn loss(&mut self) -> CResult<Tensor> {
        use std::io::Write;
        self.tokenizer.clear();
        //        let dtype = self.model.dtype();
        let dtype = DType::F32;

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
        let mut next_logits: Option<Tensor> = None;
        let label_token = tokens.last().unwrap();
        let mut pred_tokens = 0;
        println!("tokens len: {}", tokens.len());
        if tokens.len() > 2 {
            pred_tokens = tokens.len() - 1;
        } else {
            pred_tokens = 1;
        }
        let token_labels = tokens.clone();
        let label_tokens = token_labels
            .iter()
            .rev()
            .take(pred_tokens)
            .collect::<Vec<&u32>>();
        let mut prediction_tokens = vec![];
        for &t in tokens.iter().take(tokens.len() - pred_tokens) {
            prediction_tokens.push(t);
            //println!("forward propagated mamba!");
            //
            //            next_logits = Some(logits);
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("|{t}|")
            }
        }

        std::io::stdout().flush()?;

        self.tokenizer.clear();
        let mut loss = Tensor::new(&[0.0], &self.device)?
            .to_dtype(dtype)?
            .squeeze(0)?;
        //            .squeeze(0)?;
        let mut iter = 0 as f32;
        for label_token in label_tokens.into_iter().rev() {
            let input = Tensor::new(prediction_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input.clone())?.squeeze(0)?;

            iter += 1.;
            let itert = Tensor::new(&[iter], &self.device)?.squeeze(0);
            //            let logits = match next_logits.as_ref() {
            //                Some(logits) => logits.to_dtype(dtype)?,
            //                None => panic!("cannot train on an empty prompt"),
            //            };
            let pred_logits = logits.clone().to_dtype(dtype)?;

            let next_token = self
                .logits_processor
                .sample(&pred_logits.squeeze(0)?.to_dtype(dtype)?)?;
            let next_prediction_token = self.tokenizer.next_token(next_token)?;
            if let Some(t) = next_prediction_token.clone() {
                print!("[{t}]");
            } else {
                print!("[-]");
            }

            let label = Tensor::new(&[label_token.clone()], &self.device)?;

            loss = (loss + candle_nn::loss::cross_entropy(&pred_logits, &label)?)?;
            //loss =  candle_nn::loss::cross_entropy(&pred_logits, &label)?;

            //            loss = (loss
            //                + itert
            //                    * candle_nn::loss::cross_entropy(&pred_logits, &label)?
            //                        .div(&orig_token_len.clone())?)?;

            //            let input = Tensor::new(&[next_token.clone()], &self.device)?;
            prediction_tokens.push(next_token.clone());
            //            self.model.input = Some(input.clone());
            //            let logits = self.model.forward(&input.clone())?;
            next_logits = Some(logits);
        }

        tokens.clear();
        self.tokenizer.clear();
        //TODO: EXTRACT TO LBFGS
        if loss.as_ref().clone().to_scalar::<f32>()?.is_nan() {
            println!("nan");
            loss = Tensor::new(&[f32::MAX], &self.device)?
                .to_dtype(dtype)
                .unwrap();
            //                .squeeze(0)?;
        }
        //TODO: EXTRACT TO LBFGS
        return Ok(loss);
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
            args.temperature.unwrap_or(1.),
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
        println!("retrieved files: {:?}", filenames);
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let start = std::time::Instant::now();

        let read_config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let config: Config = Config {
            d_model: 40,
            n_layer: 40,
            vocab_size: read_config.vocab_size,
            pad_vocab_size_multiple: read_config.pad_vocab_size_multiple,
        };

        let device = candle_examples::device(args.cpu)?;
        let dtype = DType::from_str(&args.dtype)?;

        let mut vm = VarMap::new();
        let mut vb = VarBuilder::from_varmap(&vm, dtype, &device);

        let mut model = Model::new(&read_config, vb.pp("backbone"))?;
        vm.load_multi(&filenames)?;

        println!("loaded the model in {:?}", start.elapsed());

        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Ok(Self {
            model,
            vars: vm,
            data: vec![],
            config: read_config,
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

    fn train(
        &mut self,
        mut lbfgs_state: Option<lbfgs::lbfgs_state>,
        iter: usize,
        mut converged: bool,
    ) -> Result<(Tensor, Option<lbfgs::lbfgs_state>)> {
        let mut vars = self.vars.all_vars();
        let mut loss = self.loss().unwrap();
        let device = self.clone().device;

        println!("begin training..");

        let params = lbfgs::ParamsLBFGS {
            history_size: 10,
            line_search: Some(lbfgs::LineSearch::StrongWolfe(1e-4, 0.9, 1e-13)),
            weight_decay: Some(1e-9),
            ..Default::default()
        };
        let mut reset = false;
        let mut prev_loss = f32::MAX;
        let mut prev_loss = Tensor::new(&[prev_loss], &device)?;

        let mut lbfgs_opt = lbfgs::Lbfgs::new(self.vars.all_vars(), params, self)?;
        if lbfgs_state.is_some() {
            println!("loading Hessian..");
            lbfgs_opt.load_state(lbfgs_state.unwrap());
        }
        for i in 0..1 {
            let res = lbfgs_opt.backward_step(&mut loss).unwrap();
            match res {
                ModelOutcome::Converged(new_loss, _) => {
                    loss = new_loss.clone();

                    reset = true;
                    break;
                }
                ModelOutcome::GradConverged(new_loss, _) => {
                    loss = new_loss.clone();
                    reset = true;

                    println!("GRAD CONVERGED");

                    break;
                }
                ModelOutcome::Stepped(new_loss, _) => {
                    prev_loss = loss.clone();
                    print!(
                        "\t \t prev loss: {}  ->  ",
                        loss.clone().to_scalar::<f32>()?
                    );
                    loss = new_loss.clone();
                    println!("\t \t cur loss: {}", loss.clone().to_scalar::<f32>()?);
                }

                _ => panic!("unexpected outcome"),
            }

            if (loss.clone().to_scalar::<f32>()? <= 0.0) {
                lbfgs_opt.next_grad = None;
                lbfgs_opt.last_grad = None;
                lbfgs_opt.last_step = None;
                break;
            }
        }
        println!("\t \t \t final loss: {}", loss.clone());

        lbfgs_opt.next_grad = None;
        println!("s_hist: {}", lbfgs_opt.s_hist.len());
        converged = false;

        //TODO: EXTRACT TO LBFGS
        if reset {
            println!("RESET");
            let mut rng = rand::thread_rng();

            let mut shuffled_iterable = vars;
            let num_entries = (shuffled_iterable.len() as f32 * 0.9) as usize;
            shuffled_iterable.shuffle(&mut rng);

            shuffled_iterable
                .into_iter()
                .take(num_entries)
                .for_each(|mut x| {
                    let mut vec_tens = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    for i in 0..vec_tens.len() * 0.1 as usize {
                        let first_idx = rng.gen_range(0..=vec_tens.len() - 1);

                        let random_float: f32 = rng.gen_range(-1.0..=1.0);

                        vec_tens[first_idx] = random_float;
                    }
                    let sparse = &Tensor::from_vec(vec_tens, x.shape(), &device).unwrap();

                    x.set(sparse);
                    x = Var::from_tensor(sparse).unwrap();
                    println!("shape: {:?}", x.as_tensor());
                });

            lbfgs_opt.s_hist.clear();
            lbfgs_opt.last_grad = None;
            lbfgs_opt.last_step = None;
            lbfgs_opt.next_grad = None;
            lbfgs_opt.first = true;

            println!("-------------------------------------CLEARING Stuck Hessian {}..----------------------------------------------------", 0.);
        }
        //TODO: EXTRACT TO LBFGS
        let mut lbfgs_state = lbfgs_opt.save_state();

        Ok((loss.copy()?, Some(lbfgs_state)))
    }

    fn run_trained(&mut self) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let dtype = DType::F32;
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
            //            self.model.input = Some(input);
            let logits = self.model.forward(&input.clone())?;

            next_logits = Some(logits);
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let start_gen = std::time::Instant::now();

        let logits = match next_logits.as_ref() {
            Some(logits) => logits,
            None => anyhow::bail!("cannot work on an empty prompt"),
        };
        let logits = logits.squeeze(0)?.to_dtype(dtype)?;

        let next_token = self.logits_processor.sample(&logits)?;
        tokens.push(next_token);
        generated_tokens += 1;

        if let Some(t) = self.tokenizer.next_token(next_token)? {
            print!("\t {t}");
            std::io::stdout().flush()?;
        }

        if let Some(t) = self.tokenizer.next_token(next_token)? {
            print!("\t {t}");
            std::io::stdout().flush()?;
        }
        let input = Tensor::new(&[next_token], &self.device)?;
        //        self.model.input = Some(input);
        next_logits = Some(self.model.forward(&input.clone()).unwrap());

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
        self.tokenizer.clear();
        let dtype = DType::F32;
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
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
        for &t in tokens.iter() {
            let input = Tensor::new(&[t], &self.device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let logits = self
                .model
                .forward(&input.clone())
                .unwrap()
                .squeeze(0)
                .unwrap();

            next_logits = Some(logits);
            if let Some(t) = self.tokenizer.next_token(t).unwrap() {
                print!("{t}")
            }
        }
        std::io::stdout().flush().unwrap();
        self.tokenizer.clear();

        let start_gen = std::time::Instant::now();
        for _ in 0..sample_len {
            let logits = match next_logits.as_ref() {
                Some(logits) => logits,
                None => anyhow::bail!("cannot work on an empty prompt"),
            };
            let logits = logits.squeeze(0).unwrap().to_dtype(dtype).unwrap();
            //            let logits = if self.repeat_penalty == 1. {
            //                logits
            //            } else {
            //                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
            //                candle_transformers::utils::apply_repeat_penalty(
            //                    &logits,
            //                    self.repeat_penalty,
            //                    &tokens[start_at..],
            //                ).unwrap()
            //            };
            let next_token = self.logits_processor.sample(&logits).unwrap();
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                println!("EOS");
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token).unwrap() {
                print!("{t}");
                std::io::stdout().flush().unwrap();
            } else {
                println!("[{next_token}]");
            }

            let input = Tensor::new(&[next_token], &self.device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            next_logits = Some(
                self.model
                    .forward(&input.clone())
                    .unwrap()
                    .squeeze(0)
                    .unwrap(),
            );
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg).unwrap() {
            print!("{rest}");
        }
        std::io::stdout().flush().unwrap();
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
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    prompt: String,

    #[arg(long)]
    temperature: Option<f64>,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long, default_value_t = 299792458)]
    seed: u64,

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

    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

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
        .chunks(500)
        .map(|chunk| String::from_utf8_lossy(chunk).into_owned())
        .collect()
}

fn main() -> Result<()> {
    let mut prompts = lines_from_file("candle-examples/examples/mamba-trainer/RustLang.txt");
    let mut opt_state = None;
    let mut i = 1;
    let mut converged = false;
    let mut learner = Trainer::new()?;
    learner.run("mamba is the", 20);
    for mut j in 1..11 * prompts.len() {
        let mut learner = Trainer::new()?;
        learner.add_dataset(prompts.clone());
        learner.data.rotate_left(j);
        opt_state = learner.train(opt_state.clone(), 10, converged)?.1;
        i = j;
        learner.run("mamba is a", 20);
        println!("done training!");
        learner.run("Rust is  ", 20);
    }

    Ok(())
}
