"""Diagnose why gripper commands return 200 but the device does not move.

Checks the controller's *control-mode*: if it's `routine_editor`, the API
accepts commands (HTTP 200) but the controller ignores them — the iPad / web
Routine Editor holds control. Switching the mode to `api` is required for the
API to actually drive the arm/gripper.
"""
import os
import pathlib

from standardbots import StandardBotsRobot
from standardbots.auto_generated import models
from standardbots.auto_generated.models import (
    GripperKindEnum,
    RobotControlMode,
    RobotControlModeEnum,
)


def _load_env():
    here = pathlib.Path(__file__).resolve().parent
    for p in (here / ".env", here.parent / ".env"):
        if p.is_file():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break


_load_env()
URL = os.environ.get("ROBOT_URL", "https://cb2347.sb.app")
TOKEN = os.environ.get("ROBOT_TOKEN", "")   # set in .env

sdk = StandardBotsRobot(
    url=URL,
    token=TOKEN,
    robot_kind=StandardBotsRobot.RobotKind.Live,
)


def show(label, resp):
    raw = resp.response
    body = ""
    try:
        if raw is not None and raw.data:
            body = raw.data.decode("utf-8")
    except Exception as e:
        body = f"<decode error: {e}>"
    print(f"--- {label} ---")
    print(f"  status: {resp.status}")
    print(f"  parsed: {resp.data!r}")
    print(f"  body:   {body!r}")


print(f"URL:        {URL}")
print(f"robot_kind: {sdk._request_manager.robot_kind.value}")
print()

with sdk.connection():
    show("control-mode (before)", sdk.status.control.get_configuration_state_control())
    show("bot_identity",          sdk.general.bot_identity.bot_identity())
    show("brakes_state",          sdk.movement.brakes.get_brakes_state())
    show("gripper_config",        sdk.equipment.get_gripper_configuration())

    print("\n=== claiming API control ===")
    show("set control-mode=api",
         sdk.status.control.set_configuration_control_state(
             body=RobotControlMode(kind=RobotControlModeEnum.Api),
         ))
    show("control-mode (after)", sdk.status.control.get_configuration_state_control())

    def grip(width_mm: float):
        return sdk.equipment.control_gripper(
            body=models.GripperCommandRequest(
                kind=GripperKindEnum.Onrobot2Fg7,
                onrobot_2fg7=models.OnRobot2FG7GripperCommandRequest(
                    grip_direction=models.LinearGripDirectionEnum.External,
                    control_kind=models.OnRobot2FG7ControlKindEnum.Move,
                    target_grip_width=models.LinearUnit(
                        unit_kind=models.LinearUnitKind.Millimeters,
                        value=float(width_mm),
                    ),
                    target_force=models.ForceUnit(
                        unit_kind=models.ForceUnitKind.Newtons, value=20.0,
                    ),
                ),
            ),
        )

    print("\n=== sending OPEN (60mm) ===")
    show("control_gripper(open)", grip(60.0))
    print("\n=== sending CLOSE (35mm) ===")
    show("control_gripper(close)", grip(35.0))
