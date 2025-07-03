import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Image
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge


class Lane(Node):
    def __init__(self):
        super().__init__('color_checker')

        img_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.img_publisher = self.create_publisher(Image, 'img_data', img_qos_profile)

        self.imu_subscriber = self.create_subscription(
            Float32MultiArray,
            'imu_data',
            self.imu_msg_sampling,
            QoSProfile(depth=2)
        )

        # 중심 좌표 퍼블리셔
        self.center_publisher = self.create_publisher(Float32, 'center_x', 10)

        self.capture_timer = self.create_timer(1/15, self.image_capture)
        self.process_timer = self.create_timer(1/15, self.image_processing)

        self.U_detection_threshold = 130
        self.img_size_x = 848
        self.img_size_y = 480
        self.depth_size_x = 848
        self.depth_size_y = 480

        self.get_logger().info("Trying to access RealSense camera")

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.img_size_x, self.img_size_y, rs.format.bgr8, 15)
        self.config.enable_stream(rs.stream.depth, self.depth_size_x, self.depth_size_y, rs.format.z16, 15)

        depth_profile = self.pipeline.start(self.config)
        depth_sensor = depth_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.clipping_distance = 1 / self.depth_scale

        self.align = rs.align(rs.stream.color)
        self.hole_filling_filter = rs.hole_filling_filter()

        self.get_logger().info("RealSense access complete")

        self.cvbrid = CvBridge()

        self.L_sum = 0
        self.R_sum = 0
        self.center_x = int(self.img_size_x / 2)

        self.color_ROI = np.zeros((int(self.img_size_y * 0.2), int(self.img_size_x * 0.9), 3), dtype=np.uint8)

        self.get_logger().info("Initialization complete")

    def image_capture(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        self.aligned_depth_frame = aligned_frames.get_depth_frame()
        self.filled_depth_frame = self.hole_filling_filter.process(self.aligned_depth_frame)

        self.depth_intrinsics = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        self.depth_img = np.asanyarray(self.filled_depth_frame.get_data())
        self.color_img = np.asanyarray(color_frame.get_data())

    def yuv_detection(self, img):
        y, x, c = img.shape

        gaussian = cv2.GaussianBlur(img, (3, 3), 1)
        yuv_img = cv2.cvtColor(gaussian, cv2.COLOR_BGR2YUV)
        _, U_img, _ = cv2.split(yuv_img)
        _, U_img_treated = cv2.threshold(U_img, self.U_detection_threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(U_img_treated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        filtered = np.zeros_like(img)
        center = None
        left = right = 0

        if max_contour is not None:
            max_contour_mask = np.zeros_like(U_img_treated)
            cv2.drawContours(max_contour_mask, [max_contour], -1, 255, thickness=cv2.FILLED)

            filtered = cv2.bitwise_and(img, img, mask=max_contour_mask)
            cv2.imshow("filter", filtered)
            cv2.waitKey(1)

            x_rect, y_rect, w, h = cv2.boundingRect(max_contour)
            left = x_rect
            right = x_rect + w
            center = int((left + right) / 2)

            roi_center_x = int(x / 2)
            angle_offset = center - roi_center_x
            alpha = 0.2
            target_center_x = int((self.img_size_x * 0.05)) + center
            self.center_x = int((1 - alpha) * self.center_x + alpha * target_center_x)

            # 시각화
            cv2.line(img, (left, 0), (left, y), (0, 255, 255), 1)
            cv2.line(img, (right, 0), (right, y), (0, 255, 255), 1)
            cv2.drawContours(img, [max_contour], -1, (0, 0, 255), 2)

            self.get_logger().info(f"Angle Offset (center deviation): {angle_offset}")
            self.get_logger().info(f"Lsum: {self.L_sum}, Rsum: {self.R_sum}")

        self.img_publisher.publish(self.cvbrid.cv2_to_imgmsg(filtered))

        histogram = np.sum(U_img_treated, axis=0)
        midpoint = int(x / 2)
        L_histo = histogram[:midpoint]
        R_histo = histogram[midpoint:]
        L_sum = int(np.sum(L_histo) / 255)
        R_sum = int(np.sum(R_histo) / 255)  

        return L_sum, midpoint, R_sum

    def image_processing(self):
        y1 = int(self.img_size_y * 0.7)
        y2 = int(self.img_size_y * 0.9)
        x1 = int(self.img_size_x * 0.05)
        x2 = int(self.img_size_x * 0.95)

        self.color_ROI = self.color_img[y1:y2, x1:x2]
        l_sum, midpoint, r_sum = self.yuv_detection(self.color_ROI)
        self.L_sum = l_sum
        self.R_sum = r_sum

        # 전체 이미지에 UI 출력
        cv2.rectangle(self.color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(self.color_img, (int(self.img_size_x / 2), y1), (int(self.img_size_x / 2), y2), (0, 225, 0), 2)
        cv2.line(self.color_img, (self.center_x, y1), (self.center_x, y2), (255, 0, 0), 2)

        cv2.putText(self.color_img, f'L : {self.L_sum}   R : {self.R_sum}', (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        # 중심 좌표 퍼블리시
        center_msg = Float32()
        center_msg.data = float(self.center_x)
        self.center_publisher.publish(center_msg)

        cv2.imshow("Color Image", self.color_img)
        cv2.imshow("ROI View", self.color_ROI)
        cv2.waitKey(1)

    def imu_msg_sampling(self, msg):
        if len(msg.data) >= 3:
            self.roll = msg.data[0]
            self.pitch = msg.data[1]
            self.yaw = msg.data[2]

            self.get_logger().info(f'IMU Received - Roll: {self.roll:.2f}, Pitch: {self.pitch:.2f}, Yaw: {self.yaw:.2f}')


def main(args=None):
    rclpy.init(args=args)
    node = Lane()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
