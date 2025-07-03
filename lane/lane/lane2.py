import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # QoS 설정
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                 history=HistoryPolicy.KEEP_LAST,
                                 depth=1)

        # 이미지 퍼블리셔
        self.img_publisher = self.create_publisher(Image, 'img_data', qos_profile)

        # IMU 구독
        self.imu_subscriber = self.create_subscription(
            Float32MultiArray,
            'imu_data',
            self.imu_callback,
            QoSProfile(depth=2)
        )

        # RealSense 카메라 초기화
        self.img_size_x = 848
        self.img_size_y = 480
        self.U_threshold = 130

        self.cvbrid = CvBridge()

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.img_size_x, self.img_size_y, rs.format.bgr8, 15)
        self.config.enable_stream(rs.stream.depth, self.img_size_x, self.img_size_y, rs.format.z16, 15)
        profile = self.pipeline.start(self.config)

        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.clipping_distance = 2.5 / self.depth_scale  # 2.5m

        # 필터들
        self.hole_filling = rs.hole_filling_filter()
        self.temporal = rs.temporal_filter()
        self.spatial = rs.spatial_filter()

        # 타이머
        self.create_timer(1/15, self.image_capture)
        self.create_timer(1/15, self.image_processing)

        # 내부 상태 초기화
        self.color_img = np.zeros((self.img_size_y, self.img_size_x, 3), dtype=np.uint8)
        self.depth_img = np.zeros((self.img_size_y, self.img_size_x), dtype=np.uint16)
        self.center_x = self.img_size_x // 2
        self.L_sum = 0
        self.R_sum = 0

        self.get_logger().info("Lane Detection Node initialized")

    def image_capture(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()

        # 필터 적용
        depth = self.spatial.process(depth)
        depth = self.temporal.process(depth)
        depth = self.hole_filling.process(depth)

        self.color_img = np.asanyarray(color.get_data())
        self.depth_img = np.asanyarray(depth.get_data())

    def image_processing(self):
        x1 = int(self.img_size_x * 0.05)
        x2 = int(self.img_size_x * 0.95)
        y1 = int(self.img_size_y * 0.7)
        y2 = int(self.img_size_y * 0.9)

        roi = self.color_img[y1:y2, x1:x2]
        roi_depth = self.depth_img[y1:y2, x1:x2]

        # Gaussian + YUV 변환
        blur = cv2.GaussianBlur(roi, (3, 3), 1)
        yuv = cv2.cvtColor(blur, cv2.COLOR_BGR2YUV)
        _, U, _ = cv2.split(yuv)

        ret, U_thresh = cv2.threshold(U, self.U_threshold, 255, cv2.THRESH_BINARY)

        mask = np.zeros_like(U_thresh)
        filtered = np.zeros_like(roi)
        max_contour = None

        if ret:
            contours, _ = cv2.findContours(U_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_contour = contour

            if max_contour is not None:
                cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)

                # Depth 마스크 적용 (0.3m ~ 2.5m)
                depth_mask = np.where((roi_depth > 300) & (roi_depth < 2500), 255, 0).astype(np.uint8)
                final_mask = cv2.bitwise_and(mask, depth_mask)

                filtered = cv2.bitwise_and(roi, roi, mask=final_mask)

                mid = mask.shape[1] // 2
                self.L_sum = cv2.countNonZero(final_mask[:, :mid])
                self.R_sum = cv2.countNonZero(final_mask[:, mid:])

                # 중심 계산 및 EMA 보정
                x, y, w, h = cv2.boundingRect(max_contour)
                contour_center = x + w // 2
                roi_center = (x2 - x1) // 2
                angle_offset = contour_center - roi_center

                alpha = 0.2
                self.center_x = int((1 - alpha) * self.center_x + alpha * (x1 + contour_center))

                # 로그 출력
                self.get_logger().info(f"Angle Offset: {angle_offset}, L_sum: {self.L_sum}, R_sum: {self.R_sum}")

                # 시각화
                cv2.line(self.color_img, (self.center_x, y1), (self.center_x, y2), (255, 0, 0), 4)
                cv2.line(roi, (roi_center, 0), (roi_center, y2 - y1), (0, 255, 0), 2)
                cv2.line(roi, (x, 0), (x, y2 - y1), (0, 255, 255), 2)
                cv2.line(roi, (x + w, 0), (x + w, y2 - y1), (0, 255, 255), 2)
                cv2.drawContours(roi, [max_contour], -1, (0, 0, 255), 2)

        # 이미지 퍼블리시 및 디버깅
        self.img_publisher.publish(self.cvbrid.cv2_to_imgmsg(filtered, encoding="bgr8"))
        cv2.rectangle(self.color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(self.color_img, f'L: {self.L_sum}  R: {self.R_sum}', (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        cv2.imshow("Color Image", self.color_img)
        cv2.imshow("ROI", roi)
        cv2.imshow("Filtered", filtered)
        cv2.waitKey(1)

    def imu_callback(self, msg):
        # 추후 처리 로직 삽입 가능
        pass


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
