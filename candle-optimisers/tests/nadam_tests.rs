use candle::test_utils::{to_vec0_round, to_vec2_round};

use anyhow::Result;
use candle::{Device, Tensor, Var};
use candle_nn::{Linear, Module, Optimizer};
use candle_optimisers::{
    nadam::{NAdam, ParamsNAdam},
    Decay,
};

/* The results of this test have been checked against the following PyTorch code.
    import torch
    from torch import optim

    w_gen = torch.tensor([[3., 1.]])
    b_gen = torch.tensor([-2.])

    sample_xs = torch.tensor([[2., 1.], [7., 4.], [-4., 12.], [5., 8.]])
    sample_ys = sample_xs.matmul(w_gen.t()) + b_gen

    m = torch.nn.Linear(2, 1)
    with torch.no_grad():
        m.weight.zero_()
        m.bias.zero_()
    optimiser = optim.NAdam(m.parameters())
    # optimiser.zero_grad()
    for _step in range(100):
        optimiser.zero_grad()
        ys = m(sample_xs)
        loss = ((ys - sample_ys)**2).sum()
        loss.backward()
        optimiser.step()
        # print("Optimizer state begin")
        # print(optimiser.state)
        # print("Optimizer state end")
    print(m.weight)
    print(m.bias)
*/
#[test]
fn nadam_test() -> Result<()> {
    // Generate some linear data, y = 3.x1 + x2 - 2.
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;

    let params = ParamsNAdam::default();
    // Now use backprop to run a linear regression between samples and get the coefficients back.
    let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
    let b = Var::new(0f32, &Device::Cpu)?;
    let mut n_sgd = NAdam::new(vec![w.clone(), b.clone()], params)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..100 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        n_sgd.backward_step(&loss)?;
    }
    assert_eq!(to_vec2_round(&w, 4)?, &[[0.1897, 0.1837]]);
    assert_eq!(to_vec0_round(&b, 4)?, 0.1864);
    Ok(())
}

/* The results of this test have been checked against the following PyTorch code.
    import torch
    from torch import optim

    w_gen = torch.tensor([[3., 1.]])
    b_gen = torch.tensor([-2.])

    sample_xs = torch.tensor([[2., 1.], [7., 4.], [-4., 12.], [5., 8.]])
    sample_ys = sample_xs.matmul(w_gen.t()) + b_gen

    m = torch.nn.Linear(2, 1)
    with torch.no_grad():
        m.weight.zero_()
        m.bias.zero_()
    optimiser = optim.NAdam(m.parameters(), weight_decay = 0.6)
    # optimiser.zero_grad()
    for _step in range(100):
        optimiser.zero_grad()
        ys = m(sample_xs)
        loss = ((ys - sample_ys)**2).sum()
        loss.backward()
        optimiser.step()
        # print("Optimizer state begin")
        # print(optimiser.state)
        # print("Optimizer state end")
    print(m.weight)
    print(m.bias)
*/
#[test]
fn nadam_weight_decay_test() -> Result<()> {
    // Generate some linear data, y = 3.x1 + x2 - 2.
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;

    let params = ParamsNAdam {
        weight_decay: Some(Decay::WeightDecay(0.6)),
        ..Default::default()
    };
    // Now use backprop to run a linear regression between samples and get the coefficients back.
    let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
    let b = Var::new(0f32, &Device::Cpu)?;
    let mut n_sgd = NAdam::new(vec![w.clone(), b.clone()], params)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..100 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        n_sgd.backward_step(&loss)?;
    }
    assert_eq!(to_vec2_round(&w, 4)?, &[[0.1897, 0.1837]]);
    assert_eq!(to_vec0_round(&b, 4)?, 0.1863);
    Ok(())
}

/* The results of this test have been checked against the following PyTorch code.
    import torch
    from torch import optim

    w_gen = torch.tensor([[3., 1.]])
    b_gen = torch.tensor([-2.])

    sample_xs = torch.tensor([[2., 1.], [7., 4.], [-4., 12.], [5., 8.]])
    sample_ys = sample_xs.matmul(w_gen.t()) + b_gen

    m = torch.nn.Linear(2, 1)
    with torch.no_grad():
        m.weight.zero_()
        m.bias.zero_()
    optimiser = optim.NAdam(m.parameters(), weight_decay = 0.6, decoupled_weight_decay=True)
    # optimiser.zero_grad()
    for _step in range(100):
        optimiser.zero_grad()
        ys = m(sample_xs)
        loss = ((ys - sample_ys)**2).sum()
        loss.backward()
        optimiser.step()
        # print("Optimizer state begin")
        # print(optimiser.state)
        # print("Optimizer state end")
    print(m.weight)
    print(m.bias)
*/
#[test]
fn nadam_decoupled_weight_decay_test() -> Result<()> {
    // Generate some linear data, y = 3.x1 + x2 - 2.
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;

    let params = ParamsNAdam {
        weight_decay: Some(Decay::DecoupledWeightDecay(0.6)),
        ..Default::default()
    };
    // Now use backprop to run a linear regression between samples and get the coefficients back.
    let w = Var::new(&[[0f32, 0.]], &Device::Cpu)?;
    let b = Var::new(0f32, &Device::Cpu)?;
    let mut n_sgd = NAdam::new(vec![w.clone(), b.clone()], params)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    for _step in 0..100 {
        let ys = lin.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        n_sgd.backward_step(&loss)?;
    }
    assert_eq!(to_vec2_round(&w, 4)?, &[[0.1792, 0.1737]]);
    assert_eq!(to_vec0_round(&b, 4)?, 0.1762);
    Ok(())
}