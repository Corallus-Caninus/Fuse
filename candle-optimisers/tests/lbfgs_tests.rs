use anyhow::Result;
use candle::test_utils::to_vec2_round;
use candle::{DType, Device, Result as CResult, Tensor};
use candle_optimisers::lbfgs::{GradConv, Lbfgs, LineSearch, ParamsLBFGS, StepConv};
use candle_optimisers::{LossOptimizer, ModelOutcome, Trainable};

/*
These tests all use the 2D Rosenbrock function as a test function for the optimisers. This has minimum 0 at (1, 1)
*/

#[derive(Debug, Clone)]
pub struct RosenbrockModel {
    x_pos: candle::Var,
    y_pos: candle::Var,
}

impl Trainable for RosenbrockModel {
    fn loss(&mut self) -> CResult<Tensor> {
        //, xs: &Tensor, ys: &Tensor
        self.forward()?.squeeze(1)?.squeeze(0)
    }
}

impl RosenbrockModel {
    fn new() -> CResult<Self> {
        let x_pos =
            candle::Var::from_tensor(&(10. * Tensor::ones((1, 1), DType::F64, &Device::Cpu)?)?)?;
        let y_pos =
            candle::Var::from_tensor(&(10. * Tensor::ones((1, 1), DType::F64, &Device::Cpu)?)?)?;
        Ok(Self { x_pos, y_pos })
    }
    fn vars(&self) -> Vec<candle::Var> {
        vec![self.x_pos.clone(), self.y_pos.clone()]
    }

    fn forward(&self) -> CResult<Tensor> {
        //, xs: &Tensor
        (1. - self.x_pos.as_tensor())?.powf(2.)?
            + 100. * (self.y_pos.as_tensor() - self.x_pos.as_tensor().powf(2.)?)?.powf(2.)?
    }
}

#[test]
fn lbfgs_test() -> Result<()> {
    let params = ParamsLBFGS {
        lr: 1.,
        ..Default::default()
    };

    let model = &mut RosenbrockModel::new()?;

    let mut lbfgs = Lbfgs::new(model.vars(), params, model.clone())?;
    let mut loss = model.loss()?;

    for _step in 0..500 {
        // println!("\nstart step {}", step);
        // for v in model.vars() {
        //     println!("{}", v);
        // }
        let res = lbfgs.backward_step(&loss)?; //&sample_xs, &sample_ys
                                               // println!("end step {}", _step);
        match res {
            ModelOutcome::Converged(_, _) => break,
            ModelOutcome::Stepped(new_loss, _) => loss = new_loss,
            // _ => panic!("unexpected outcome"),
        }
    }

    for v in model.vars() {
        // println!("{}", v);
        assert_eq!(to_vec2_round(&v.to_dtype(DType::F32)?, 4)?, &[[1.0000]]);
    }

    // println!("{:?}", lbfgs);
    // panic!("deliberate panic");

    Ok(())
}

#[test]
fn lbfgs_test_strong_wolfe() -> Result<()> {
    let params = ParamsLBFGS {
        lr: 1.,
        line_search: Some(LineSearch::StrongWolfe(1e-4, 0.9, 1e-9)),
        ..Default::default()
    };

    let model = &mut RosenbrockModel::new()?;

    let mut lbfgs = Lbfgs::new(model.vars(), params, model.clone())?;
    let mut loss = model.loss()?;

    for _step in 0..500 {
        // println!("\nstart step {}", step);
        // for v in model.vars() {
        //     println!("{}", v);
        // }
        let res = lbfgs.backward_step(&loss)?; //&sample_xs, &sample_ys
                                               // println!("end step {}", _step);
        match res {
            ModelOutcome::Converged(_, _) => break,
            ModelOutcome::Stepped(new_loss, _) => loss = new_loss,
            // _ => panic!("unexpected outcome"),
        }
    }

    for v in model.vars() {
        // println!("{}", v);
        assert_eq!(to_vec2_round(&v.to_dtype(DType::F32)?, 4)?, &[[1.0000]]);
    }

    // println!("{:?}", lbfgs);
    // panic!("deliberate panic");

    Ok(())
}

#[test]
fn lbfgs_rms_grad_test() -> Result<()> {
    let params = ParamsLBFGS {
        lr: 1.,
        grad_conv: GradConv::RMSForce(1e-6),
        ..Default::default()
    };

    let model = &mut RosenbrockModel::new()?;

    let mut lbfgs = Lbfgs::new(model.vars(), params, model.clone())?;
    let mut loss = model.loss()?;

    for _step in 0..500 {
        // println!("\nstart step {}", step);
        // for v in model.vars() {
        //     println!("{}", v);
        // }
        let res = lbfgs.backward_step(&loss)?; //&sample_xs, &sample_ys
                                               // println!("end step {}", _step);
        match res {
            ModelOutcome::Converged(_, _) => break,
            ModelOutcome::Stepped(new_loss, _) => loss = new_loss,
            // _ => panic!("unexpected outcome"),
        }
    }

    for v in model.vars() {
        // println!("{}", v);
        assert_eq!(to_vec2_round(&v.to_dtype(DType::F32)?, 4)?, &[[1.0000]]);
    }

    // println!("{:?}", lbfgs);
    // panic!("deliberate panic");

    Ok(())
}

#[test]
fn lbfgs_rms_step_test() -> Result<()> {
    let params = ParamsLBFGS {
        lr: 1.,
        grad_conv: GradConv::RMSForce(0.),
        step_conv: StepConv::RMSStep(1e-7),
        ..Default::default()
    };

    let model = &mut RosenbrockModel::new()?;

    let mut lbfgs = Lbfgs::new(model.vars(), params, model.clone())?;
    let mut loss = model.loss()?;

    for _step in 0..500 {
        // println!("\nstart step {}", step);
        // for v in model.vars() {
        //     println!("{}", v);
        // }
        let res = lbfgs.backward_step(&loss)?; //&sample_xs, &sample_ys
                                               // println!("end step {}", _step);
        match res {
            ModelOutcome::Converged(_, _) => break,
            ModelOutcome::Stepped(new_loss, _) => loss = new_loss,
            // _ => panic!("unexpected outcome"),
        }
    }

    for v in model.vars() {
        // println!("{}", v);
        assert_eq!(to_vec2_round(&v.to_dtype(DType::F32)?, 4)?, &[[1.0000]]);
    }

    // println!("{:?}", lbfgs);
    // panic!("deliberate panic");

    Ok(())
}

#[test]
fn lbfgs_test_strong_wolfe_weight_decay() -> Result<()> {
    let params = ParamsLBFGS {
        lr: 1.,
        line_search: Some(LineSearch::StrongWolfe(1e-4, 0.9, 1e-9)),
        weight_decay: Some(0.1),
        ..Default::default()
    };

    let model = &mut RosenbrockModel::new()?;

    let mut lbfgs = Lbfgs::new(model.vars(), params, model.clone())?;
    let mut loss = model.loss()?;

    for _step in 0..500 {
        // println!("\nstart step {}", step);
        // for v in model.vars() {
        //     println!("{}", v);
        // }
        let res = lbfgs.backward_step(&loss)?; //&sample_xs, &sample_ys
                                               // println!("end step {}", _step);
        match res {
            ModelOutcome::Converged(_, _) => break,
            ModelOutcome::Stepped(new_loss, _) => loss = new_loss,
            // _ => panic!("unexpected outcome"),
        }
    }

    let expected = [0.8861, 0.7849]; // this should be properly checked
    for (v, e) in model.vars().iter().zip(expected) {
        // println!("{}", v);
        assert_eq!(to_vec2_round(&v.to_dtype(DType::F32)?, 4)?, &[[e]]);
    }

    // println!("{:?}", lbfgs);
    // panic!("deliberate panic");

    Ok(())
}

#[test]
fn lbfgs_test_weight_decay() -> Result<()> {
    let params = ParamsLBFGS {
        lr: 1.,
        weight_decay: Some(0.1),
        ..Default::default()
    };

    let model = &mut RosenbrockModel::new()?;

    let mut lbfgs = Lbfgs::new(model.vars(), params, model.clone())?;
    let mut loss = model.loss()?;

    for _step in 0..500 {
        // println!("\nstart step {}", step);
        // for v in model.vars() {
        //     println!("{}", v);
        // }
        let res = lbfgs.backward_step(&loss)?; //&sample_xs, &sample_ys
                                               // println!("end step {}", _step);
        match res {
            ModelOutcome::Converged(_, _) => break,
            ModelOutcome::Stepped(new_loss, _) => loss = new_loss,
            // _ => panic!("unexpected outcome"),
        }
    }

    let expected = [0.8861, 0.7849]; // this should be properly checked
    for (v, e) in model.vars().iter().zip(expected) {
        // println!("{}", v);
        assert_eq!(to_vec2_round(&v.to_dtype(DType::F32)?, 4)?, &[[e]]);
    }

    // println!("{:?}", lbfgs);
    // panic!("deliberate panic");

    Ok(())
}
