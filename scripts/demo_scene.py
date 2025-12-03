import argparse
from isaaclab.app import AppLauncher

# 1. Isaac Sim 실행 설정
parser = argparse.ArgumentParser(description="Simple Teleop script for Precise Manip environment.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------
import torch
import numpy as np
import zmq
import pickle
from isaaclab.envs import ManagerBasedRLEnv

# 사용자 환경 설정 임포트
from precise_manip.tasks.manager_based.precise_manip.precise_manip_env_cfg import PreciseManipEnvCfg

def main():
    # 2. 환경 설정 및 생성
    env_cfg = PreciseManipEnvCfg()
    env_cfg.scene.num_envs = 1 
    env_cfg.scene.env_spacing = 5.0
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 3. ZMQ 서버 시작
    print("[INFO] Initializing ZMQ Server...")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    try:
        socket.bind("tcp://*:6001")
        print("[INFO] ZMQ Server Ready on port 6001.")
    except zmq.ZMQError as e:
        print(f"[ERROR] Port bind failed: {e}")
        return

    # 4. 시뮬레이션 시작
    print("[INFO] Simulation started (Direct Control Mode).")
    obs, _ = env.reset()
    
    # Action 차원 설정 (7 DOF)
    action_dim = 7
    current_action = torch.zeros((env.num_envs, action_dim), device=env.device)

    # =================================================================
    # [메인 루프] 대기 없이 바로 시작
    # =================================================================
    while simulation_app.is_running():
        # (1) ZMQ 메시지 확인 (Non-blocking)
        try:
            message = socket.recv(flags=zmq.NOBLOCK)
            request = pickle.loads(message)
            method = request.get("method")

            # --- GELLO 요청 처리 ---
            if method == "command_joint_state":
                # GELLO 명령 수신 -> 즉시 반영
                target_joints_np = request["args"]["joint_state"]
                target_tensor = torch.from_numpy(target_joints_np).float().to(env.device)
                
                # 현재 액션 업데이트
                current_action[0] = target_tensor[:action_dim]
                
                socket.send(pickle.dumps("ACK"))

            elif method == "get_observations":
                # 현재 로봇 상태 반환 (필요시 실제 값 연결 가능)
                # 여기서는 연결 확인용 0 전송
                response = {
                    "joint_positions": np.zeros(7),
                    "joint_velocities": np.zeros(7),
                    "ee_pos_quat": np.zeros(7),
                    "gripper_position": np.array([0.0]),
                    "force_feedback": np.zeros(7)
                }
                socket.send(pickle.dumps(response))

            elif method == "num_dofs":
                socket.send(pickle.dumps(7))
            else:
                socket.send(pickle.dumps(None))

        except zmq.Again:
            # 메시지 없으면 이전 명령(current_action) 유지
            pass
        except Exception as e:
            print(f"[ZMQ Error] {e}")

        # (2) 환경 Step (항상 실행됨)
        # GELLO 연결 전에는 0.0 (또는 초기화 값)으로 이동하려고 할 것임
        obs, *_ = env.step(current_action)

    # 종료 처리
    print("[INFO] Simulation stopped.")
    socket.close()
    context.term()
    env.close()

if __name__ == "__main__":
    main()
