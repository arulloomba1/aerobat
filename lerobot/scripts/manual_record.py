# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script for manual recording control of robot episodes.
Press SPACE to start/stop recording, ESC to exit.
"""

import logging
import time
from dataclasses import asdict
from pprint import pformat

import rerun as rr
import torch
from pynput import keyboard
from termcolor import colored

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    ControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, has_method, init_logging, log_say
from lerobot.configs import parser


def init_keyboard_listener():
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False
    events["is_recording"] = False

    def on_press(key):
        try:
            if key == keyboard.Key.space:
                events["is_recording"] = not events["is_recording"]
                state = "STARTED" if events["is_recording"] else "STOPPED"
                print(f"Space key pressed. Recording {state}...")
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def log_control_info(robot: Robot, dt_s, fps=None):
    log_items = []
    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)


def manual_record_loop(
    robot: Robot,
    dataset: LeRobotDataset,
    events: dict,
    display_data: bool = False,
    policy = None,
    fps: int = 30,
    single_task: str = None,
):
    if not robot.is_connected:
        robot.connect()

    if policy is not None:
        device = get_safe_torch_device(policy.config.device)
        use_amp = policy.config.use_amp

    print("\nManual Recording Controls:")
    print("Press SPACE to start/stop recording")
    print("Press ESC to exit")
    print("Waiting for recording to start...")

    while True:
        start_loop_t = time.perf_counter()

        if events["stop_recording"]:
            break

        observation = robot.capture_observation()

        if policy is not None:
            with torch.inference_mode():
                pred_action = policy.select_action(observation)
                action = robot.send_action(pred_action)
                action = {"action": action}
        else:
            action = robot.teleop_step(record_data=True)[1]

        if events["is_recording"]:
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            for k, v in action.items():
                for i, vv in enumerate(v):
                    rr.log(f"sent_{k}_{i}", rr.Scalar(vv.numpy()))

            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                rr.log(key, rr.Image(observation[key].numpy()), static=True)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)


@parser.wrap()
def manual_record(cfg: ControlPipelineConfig):
    init_logging()

    if not isinstance(cfg.control, RecordControlConfig):
        raise ValueError("This script only works with record control type")

    robot = make_robot_from_config(cfg.robot)
    listener, events = init_keyboard_listener()

    # Create or load dataset
    if cfg.control.resume:
        dataset = LeRobotDataset(cfg.control.repo_id, root=cfg.control.root)
    else:
        dataset = LeRobotDataset.create(
            cfg.control.repo_id,
            cfg.control.fps,
            root=cfg.control.root,
            robot=robot,
            use_videos=cfg.control.video,
        )

    # Load policy if provided
    policy = None if cfg.control.policy is None else make_policy(cfg.control.policy, ds_meta=dataset.meta)

    try:
        manual_record_loop(
            robot=robot,
            dataset=dataset,
            events=events,
            display_data=cfg.control.display_data,
            policy=policy,
            fps=cfg.control.fps,
            single_task=cfg.control.single_task,
        )
    finally:
        robot.disconnect()
        listener.stop()

        if cfg.control.push_to_hub:
            dataset.push_to_hub(tags=cfg.control.tags, private=cfg.control.private)


if __name__ == "__main__":
    manual_record() 