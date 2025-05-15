import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray

class JoyPubTestmodel(Node):
    def __init__(self):
        super().__init__('pub_for_testmodel')
        qos_profile = QoSProfile(depth=10)
        self.joy_subscriber = self.create_subscription(
            Joy,
            'joy',
            self.joy_msg_sampling,
            qos_profile)
        
        self.joy_pub_testmodel = self.create_publisher(Float32MultiArray, 'joycmd', qos_profile)
        
        # [L수직, R수직] -> 구동모터와 플리퍼를 분리하여 데이터 전송
        self.joy_stick_data = []

    def joy_msg_sampling(self, msg):
        axes = msg.axes
        btn = msg.buttons

        #if axes[2] == -1 and axes[5] != -1:  # 구동모터 제어
        if btn[7] == 1 and btn[8] != 1:  # 구동모터 제어
            self.get_logger().info('CAR')
            #self.joy_stick_data = [1.0,axes[1], axes[3]] 
            self.joy_stick_data = [1.0,axes[1], axes[3]] 

        #elif axes[5] == -1 and axes[2] != -1:  # 플리퍼 제어
        elif btn[7] != 1 and btn[8] == 1:
            self.get_logger().info('Flipper')
            #self.joy_stick_data = [2.0,axes[1], axes[3]] 
            self.joy_stick_data = [2.0,axes[1], axes[3]] 

        else: # 정지
            self.joy_stick_data = [0.0, 0.0, 0.0] 
        
        self.joy_data_publish()

    def joy_data_publish(self):
        msg = Float32MultiArray()
        msg.data = self.joy_stick_data
        self.joy_pub_testmodel.publish(msg)
        self.get_logger().info(str(msg.data))

def main(args=None):
    rclpy.init(args=args)
    node = JoyPubTestmodel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()