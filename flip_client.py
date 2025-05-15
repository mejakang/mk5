import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Joy

from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool
from rclpy.node import Node 

class FlipperClient(Node):
    def __init__(self):
        super().__init__('flipper_client')

        self.sent = False  
        self.joy_stick_data = []

        qos_profile = QoSProfile(depth=10)

        # Joy Subscriber
        self.joy_subscriber = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,  
            qos_profile)

        # Joy Publisher
        self.joy_pub_testmodel = self.create_publisher(
            Float32MultiArray, 'joycmd', qos_profile)

        # Service Client
        self.client_joy = self.create_client(SetBool, 'flipper_control')
        while not self.client_joy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for flipper_control service...')
        self.get_logger().info('Service available.')

    def joy_callback(self, msg):
        axes = msg.axes
        btn = msg.buttons

        # Flipper Action gogo
        if btn[1] == 1 and not self.sent:
            self.send_request('STEP_ASCEND')
            self.sent = True
            return

        # CAR 모드
        elif btn[7] == 1 and btn[8] != 1:
            self.get_logger().info('CAR')
            self.joy_stick_data = [1.0, axes[1], axes[3]]

        # FLIPPER 모드
        elif btn[7] != 1 and btn[8] == 1:
            self.get_logger().info('Flipper')
            self.joy_stick_data = [2.0, axes[1], axes[3]]

        else:
            self.joy_stick_data = [0.0, 0.0, 0.0]

        self.joy_data_publish()

    def joy_data_publish(self):
        msg = Float32MultiArray()
        msg.data = self.joy_stick_data
        self.joy_pub_testmodel.publish(msg)
        self.get_logger().info(str(msg.data))

    def send_request(self, mode: str):
        req = SetBool.Request()
        if mode == 'INIT_FLIPPER':
            req.data = False
            self.get_logger().info('Flipper Initailizing')

        elif mode == 'STEP_ASCEND':
            req.data = True
            self.get_logger().info('Mission SUCCESS')
        else:
            self.get_logger().error('Invalid Mode')
            return

        future = self.client_joy.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f'Service Response: {future.result().message}')
        else:
            self.get_logger().error('Service call FAIL')

def main():
    rclpy.init()
    client = FlipperClient()
    rclpy.spin(client)
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
