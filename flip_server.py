import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

import odrive
from odrive.enums import (
    CONTROL_MODE_VELOCITY_CONTROL, INPUT_MODE_VEL_RAMP,
    AXIS_STATE_IDLE, AXIS_STATE_CLOSED_LOOP_CONTROL,
    AXIS_STATE_FULL_CALIBRATION_SEQUENCE
)

import time
from std_srvs.srv import SetBool
from std_msgs.msg import Float32MultiArray

class MobilityServer(Node):
    def __init__(self):
        super().__init__('mobility_server')

        self.get_logger().info('Connecting to ODrives...')
        try:
            self.car_drive = odrive.find_any(serial_number="336F37603433")    # 차량 
            self.flip_drive = odrive.find_any(serial_number="3755366B3331")   # 플리퍼
        except Exception as e:
            self.get_logger().error(f'ODrive connection failed: {e}')
            rclpy.shutdown()
            exit(1)

        try:
            self.calibrate_all()
        except Exception as e:
            self.get_logger().error(f'Calibration failed: {e}')
            rclpy.shutdown()
            exit(1)

        self.srv = self.create_service(SetBool, 'flipper_control', self.control_callback)

        qos_profile = QoSProfile(depth=10)
        self.motor_control_sub = self.create_subscription(
            Float32MultiArray, 
            'joycmd',
            self.subscribe_joy_message, 
            qos_profile)

    def calibrate_all(self):
        self.get_logger().info('Calibration START')

        for axis in [self.car_drive.axis0, self.car_drive.axis1]:
            axis.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
            while axis.current_state != AXIS_STATE_IDLE:
                time.sleep(0.1)
            axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        self.get_logger().info('CAR calibration complete.')

        for axis in [self.flip_drive.axis0, self.flip_drive.axis1]:
            axis.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
            while axis.current_state != AXIS_STATE_IDLE:
                time.sleep(0.1)
            axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        self.get_logger().info('FLIPPER Calibration Complete.')

    def control_callback(self, request, response):
        try:
            if request.data:  # STEP_ASCEND
                self.get_logger().info('[STEP_ASCEND]')
                self.step_ascend()
                response.success = True
                response.message = 'Executed STEP_ASCEND'
            else:  # INIT_FLIPPER
                self.get_logger().info('[INIT_FLIPPER]')
                self.init_flipper()
                response.success = True
                response.message = 'Executed INIT_FLIPPER'

        except Exception as e:
            self.get_logger().error(f'Execution error: {e}')
            response.success = False
            response.message = f'Error occurred: {e}'
        return response

    def init_flipper(self):
        self.flip_drive.axis0.encoder.set_linear_count(0)
        self.flip_drive.axis1.encoder.set_linear_count(0)
        time.sleep(0.2)

        for axis in [self.flip_drive.axis0, self.flip_drive.axis1]:
            axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            axis.controller.config.input_mode = INPUT_MODE_VEL_RAMP
            axis.controller.config.vel_ramp_rate = 5
            axis.controller.input_vel = 0

        self.get_logger().info('Flipper reset to initial position.')

    def step_ascend(self):
        for axis, vel in zip([self.flip_drive.axis0, self.flip_drive.axis1], [-2.0, 2.0]):
            axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            axis.controller.config.input_mode = INPUT_MODE_VEL_RAMP
            axis.controller.config.vel_ramp_rate = 5
            axis.controller.input_vel = vel
        self.get_logger().info('Flipper Rotating START !')
        time.sleep(5)

        for axis, vel in zip([self.car_drive.axis0, self.car_drive.axis1], [2.0, -2.0]):
            axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            axis.controller.config.input_mode = INPUT_MODE_VEL_RAMP
            axis.controller.config.vel_ramp_rate = 5
            axis.controller.input_vel = vel
        self.get_logger().info('Car driving Forward up the step')
        time.sleep(4)

        for axis in [self.flip_drive.axis0, self.flip_drive.axis1, self.car_drive.axis0, self.car_drive.axis1]:
            axis.controller.input_vel = 0
        self.get_logger().info('Complete')

    def subscribe_joy_message(self, msg):
        data = msg.data
        self.get_logger().info(f'Received joystick data: {data}')

        # 각 축 Setting = Velocity Ramp
        for axis in [self.car_drive.axis0, self.car_drive.axis1, self.flip_drive.axis0, self.flip_drive.axis1]:
            axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            axis.controller.config.input_mode = INPUT_MODE_VEL_RAMP
            axis.controller.config.vel_ramp_rate = 5

        # 기본 = 정지
        for axis in [self.car_drive.axis0, self.car_drive.axis1, self.flip_drive.axis0, self.flip_drive.axis1]:
            axis.controller.input_vel = 0

        try:
            if data[0] == 0.0:
                self.get_logger().info('STOP')
                return
            elif data[0] == 1.0:
                self.get_logger().info('CAR MODE')
                self.car_drive.axis0.controller.input_vel = data[1] * 5
                self.car_drive.axis1.controller.input_vel = -data[2] * 5
            elif data[0] == 2.0:
                self.get_logger().info('FLIPPER MODE')
                self.flip_drive.axis0.controller.input_vel = data[1] * 3
                self.flip_drive.axis1.controller.input_vel = -data[2] * 3
        except IndexError:
            self.get_logger().warn('Joystick data X')

def main():
    rclpy.init()
    node = MobilityServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
