from miniMLP.schedulers import StepLR, ExponentialLR


def test_step_lr():
    sched = StepLR(initial_lr=0.1, step_size=2, gamma=0.5)
    assert sched(0) == 0.1
    assert sched(1) == 0.1
    assert sched(2) == 0.05
    assert sched(4) == 0.025


def test_exponential_lr():
    sched = ExponentialLR(initial_lr=0.2, gamma=0.9)
    assert abs(sched(3) - 0.2 * 0.9**3) < 1e-12