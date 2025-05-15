import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import odrive
from odrive.enums import CONTROL_MODE_VELOCITY_CONTROL
from odrive.enums import INPUT_MODE_VEL_RAMP
import time
from std_msgs.msg import Float32MultiArray

class Subscriber(Node):
    def __init__(self):
        super().__init__('joy')

        self.car_drive = odrive.find_any(serial_number="336F37603433")  # 구동모터 오드라이브
        self.flip_drive = odrive.find_any(serial_number="3755366B3331")  # 플리퍼 오드라이브
        self.calibration()

        qos_profile = QoSProfile(depth=10)
        self.motor_control_sub = self.create_subscription(
            Float32MultiArray, 
            'joycmd',
            self.subscribe_joy_message,  
            qos_profile)

    def calibration(self):
        self.get_logger().info('Calibration START')
        # CAR Axis 0
        self.car_drive.axis0.requested_state = odrive.enums.AXIS_STATE_FULL_CALIBRATION_SEQUENCE
        while self.car_drive.axis0.current_state != odrive.enums.AXIS_STATE_IDLE:
            time.sleep(0.1)
     
        self.car_drive.axis0.requested_state = odrive.enums.AXIS_STATE_CLOSED_LOOP_CONTROL

        #CAR Axis 1
        self.car_drive.axis1.requested_state = odrive.enums.AXIS_STATE_FULL_CALIBRATION_SEQUENCE
        while self.car_drive.axis1.current_state != odrive.enums.AXIS_STATE_IDLE:
            time.sleep(0.1)
     
        self.car_drive.axis1.requested_state = odrive.enums.AXIS_STATE_CLOSED_LOOP_CONTROL       


        self.get_logger().info('CAR Calibration COMPLETE.')

        #Flipper  Axis0
        self.flip_drive.axis0.requested_state = odrive.enums.AXIS_STATE_FULL_CALIBRATION_SEQUENCE
        while self.flip_drive.axis0.current_state != odrive.enums.AXIS_STATE_IDLE:
            time.sleep(0.1)
        self.flip_drive.axis0.requested_state = odrive.enums.AXIS_STATE_CLOSED_LOOP_CONTROL

        #Flipper Axis1
        self.flip_drive.axis1.requested_state = odrive.enums.AXIS_STATE_FULL_CALIBRATION_SEQUENCE
        while self.flip_drive.axis1.current_state != odrive.enums.AXIS_STATE_IDLE:
            time.sleep(0.1)         
        
        self.flip_drive.axis1.requested_state = odrive.enums.AXIS_STATE_CLOSED_LOOP_CONTROL
        self.get_logger().info('Flipper COMPLETE.')

    def subscribe_joy_message(self, msg): 
        joy_stick_data = msg.data
        self.get_logger().info(f'Received joy stick data: {joy_stick_data}')

        # 구동모터 (car_drive) 설정
        self.car_drive.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        self.car_drive.axis1.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        self.car_drive.axis0.controller.config.vel_ramp_rate = 5
        self.car_drive.axis1.controller.config.vel_ramp_rate = 5
        self.car_drive.axis0.controller.config.input_mode = INPUT_MODE_VEL_RAMP
        self.car_drive.axis1.controller.config.input_mode = INPUT_MODE_VEL_RAMP

        # 플리퍼 (flip_drive) 설정
        self.flip_drive.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        self.flip_drive.axis1.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        self.flip_drive.axis0.controller.config.vel_ramp_rate = 5
        self.flip_drive.axis1.controller.config.vel_ramp_rate = 5
        self.flip_drive.axis0.controller.config.input_mode = INPUT_MODE_VEL_RAMP
        self.flip_drive.axis1.controller.config.input_mode = INPUT_MODE_VEL_RAMP

        # 기본 속도 0으로 설정
        self.car_drive.axis0.controller.input_vel = 0
        self.car_drive.axis1.controller.input_vel = 0
        self.flip_drive.axis0.controller.input_vel = 0
        self.flip_drive.axis1.controller.input_vel = 0

        # STOP 버튼 눌렀을 때 모터 정지
        if joy_stick_data[0] == 0.0:  # 정지 상태일 때
            self.car_drive.axis0.controller.input_vel = 0
            self.car_drive.axis1.controller.input_vel = 0
            self.flip_drive.axis0.controller.input_vel = 0
            self.flip_drive.axis1.controller.input_vel = 0
            self.get_logger().info('STOP')
        else:
            # CAR 모드 (joy_stick_data[0] == 0)
            if joy_stick_data[0] == 1.0:  # CAR 모드
                self.get_logger().info('CAR mode')
                self.car_drive.axis0.controller.input_vel = joy_stick_data[1] * 5  # 속도 제어
                self.car_drive.axis1.controller.input_vel = -joy_stick_data[2] * 5  # 반대 방향

            # FLIPPER 모드 (joy_stick_data[0] == 1)
            elif joy_stick_data[0] == 2.0:  # FLIPPER 모드
                self.get_logger().info('FLIPPER mode')
                self.flip_drive.axis0.controller.input_vel = joy_stick_data[1] * 3  # 플리퍼 속도
                self.flip_drive.axis1.controller.input_vel = -joy_stick_data[2] * 3  # 반대 방향

        self.get_logger().info('Control GOGO')

        time.sleep(0.1)

def main():
    rclpy.init()
    sub = Subscriber()
    rclpy.spin(sub)
    sub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
