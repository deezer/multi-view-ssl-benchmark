from typing import Any, List, Tuple, Union

import gin  # type: ignore
import numpy as np
import numpy.typing as npt


def get_cosine_scheduler(
    init_value: Union[float, int],
    final_value: Union[float, int],
    n_epochs: int,
    n_steps: int,
) -> Any:
    print("\t COSINE VALUES from {} to {}".format(init_value, final_value))
    iters = np.arange(n_epochs * n_steps)
    schedule = final_value + 0.5 * (init_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    assert len(schedule) == n_epochs * n_steps
    return schedule


def get_linear_scheduler(
    init_value: float,
    final_value: float,
    n_epochs: int,
    n_steps: int,
) -> npt.NDArray[np.float32]:
    print("\t LINEAR VALUES from {} to {}".format(init_value, final_value))
    schedule = np.linspace(init_value, final_value, n_epochs * n_steps)
    assert len(schedule) == n_epochs * n_steps
    return schedule


def get_scheduler(
    scheduler_values: List[Tuple[float, float]],
    scheduler_epochs: List[Tuple[int, int]],
    n_epochs: int,
    n_steps: int,
    scheduler_types: List[str],
) -> npt.NDArray[np.float32]:
    assert isinstance(scheduler_values, List)
    assert isinstance(scheduler_epochs, List)
    assert isinstance(scheduler_types, List)
    # assert scheduler_epochs[-1][-1] <= n_epochs
    schedulers = []
    for v, e, t in zip(scheduler_values, scheduler_epochs, scheduler_types):
        init_value, final_value = v
        init_epoch, final_epoch = e
        assert t in ["linear", "cosine"]
        print("\t {} EPOCHS from {} to {}".format(t.upper(), init_epoch, final_epoch))
        if t == "linear":
            scheduler_fn = get_linear_scheduler
        if t == "cosine":
            scheduler_fn = get_cosine_scheduler
        ee = final_epoch - init_epoch
        schedulers.append(scheduler_fn(init_value, final_value, ee, n_steps))
    if final_epoch < n_epochs:
        print("\t LINEAR EPOCHS from {} to {}".format(final_epoch, n_epochs))
        schedulers.append(get_linear_scheduler(final_value, final_value, n_epochs - final_epoch, n_steps))
    return np.concatenate(schedulers)


@gin.configurable  # type: ignore
def get_network_momentum(
    n_epochs: int,
    n_steps: int,
    momentum_teacher: float = 0.996,
    warmup_epochs: int = 0,
    warmup_init_value: float = 0.9995,
    final_value: float = 1.0,
    schedule_epochs: int = 999,
) -> npt.NDArray[np.float32]:
    """
    Args:
        momentum_teacher : base EMA parameter for teacher update.
        The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example
        use 0.9995 with batch size of 256.
    """
    print("--- NETWORK MOMENTUM ---")
    scheduler_types = ["cosine", "cosine"]
    values = [(warmup_init_value, momentum_teacher), (momentum_teacher, final_value)]
    epochs = [(0, warmup_epochs), (warmup_epochs, schedule_epochs)]
    return get_scheduler(values, epochs, n_epochs, n_steps, scheduler_types)


@gin.configurable  # type: ignore
def get_learning_rate_scheduler(
    lr: float,
    n_epochs: int,
    n_steps: int,
    min_lr: float = 1e-7,  # 5e-7
    warmup_epochs: int = 0,
    warmup_init_value: float = 1e-7,  # 1e-6
    schedule_epochs: int = 999,
) -> npt.NDArray[np.float32]:
    """
    Args:
        min_lr : Target LR at the end of optimization. We use a cosine LR
        schedule with linear warmup.
        warmup_epochs : Number of epochs for the linear learning-rate warm up.
    """
    print("--- LEARNING RATE SCHEDULE ---")
    if schedule_epochs > n_epochs:
        schedule_epochs = n_epochs
    scheduler_types = ["linear", "cosine"]
    values = [(warmup_init_value, lr), (lr, min_lr)]
    epochs = [(0, warmup_epochs), (warmup_epochs, schedule_epochs)]
    return get_scheduler(values, epochs, n_epochs, n_steps, scheduler_types)


@gin.configurable  # type: ignore
def get_weights_decay_scheduler(
    n_epochs: int,
    n_steps: int,
    weight_decay: float = 0.04,
    weight_decay_end: float = 0.4,
    schedule_epochs: int = 999,
) -> npt.NDArray[np.float32]:
    """
    Args:
        weight_decay : Initial value of the weight decay.
        weight_decay_end : Final value of the weight decay.  We use a cosine
        schedule for WD and using a larger decay by the end of training
    """
    print("--- WEIGHT DECAY SCHEDULE ---")
    if schedule_epochs > n_epochs:
        schedule_epochs = n_epochs
    scheduler_types = ["cosine"]
    values = [(weight_decay, weight_decay_end)]
    epochs = [(0, schedule_epochs)]
    return get_scheduler(values, epochs, n_epochs, n_steps, scheduler_types)


@gin.configurable  # type: ignore
def get_quantizer_schedule(
    n_epochs: int,
    temp: float = 1.0,
    temp_final: float = 1.0,
    temp_init: float = 0.25,
    warmup_epochs: int = 250,
    schedule_epochs: int = 999,
) -> npt.NDArray[np.float32]:
    print("--- quantizer LOSS SCHEDULE ---")
    if schedule_epochs > n_epochs:
        schedule_epochs = n_epochs
    scheduler_types = ["cosine", "cosine"]
    values = [(temp_init, temp), (temp, temp_final)]
    epochs = [(0, warmup_epochs), (warmup_epochs, schedule_epochs)]
    return get_scheduler(values, epochs, n_epochs, 1, scheduler_types)


def get_temp_schedule(
    n_epochs: int,
    temp: float,
    temp_final: float,
    temp_init: float,
    warmup_epochs: int,
    schedule_epochs: int,
) -> npt.NDArray[np.float32]:
    scheduler_types = ["cosine", "cosine"]
    values = [(temp_init, temp), (temp, temp_final)]
    epochs = [(0, warmup_epochs), (warmup_epochs, schedule_epochs)]
    return get_scheduler(values, epochs, n_epochs, 1, scheduler_types)


@gin.configurable  # type: ignore
def get_temp_teacher_schedule(
    n_epochs: int,
    temp: float = 0.04,
    temp_final: float = 0.04,
    temp_init: float = 0.02,
    warmup_epochs: int = 10,
    schedule_epochs: int = 999,
) -> npt.NDArray[np.float32]:
    """
    - warmup_teacher_temp: initial value for the teacher temperature:
    0.04 works well in most cases. Try decreasing it if the training loss
    does not decrease.
    we apply a warm up for the teacher temperature because
    a too high temperature makes the training instable at the beginning
    """
    print("--- TEACHER TEMPERATURE SCHEDULE ---")
    if schedule_epochs > n_epochs:
        schedule_epochs = n_epochs
    return get_temp_schedule(n_epochs, temp, temp_final, temp_init, warmup_epochs, schedule_epochs)


@gin.configurable  # type: ignore
def get_temp_student_schedule(
    n_epochs: int,
    temp: float = 0.1,
    temp_final: float = 0.1,
    temp_init: float = 0.1,
    warmup_epochs: int = 0,
    schedule_epochs: int = 999,
) -> npt.NDArray[np.float32]:
    print("--- STUDENT TEMPERATURE SCHEDULE ---")
    if schedule_epochs > n_epochs:
        schedule_epochs = n_epochs
    return get_temp_schedule(n_epochs, temp, temp_final, temp_init, warmup_epochs, schedule_epochs)
