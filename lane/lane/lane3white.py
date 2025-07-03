import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge


class Lanedecttest(Node):
    def __init__(self):
        super().__init__('line_checker')

        self.img_size_x = 848
        self.img_size_y = 480
        self.U_detection_threshold = 140

        self.cvbrid = CvBridge()

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.img_size_x, self.img_size_y, rs.format.bgr8, 15)
        self.config.enable_stream(rs.stream.depth, self.img_size_x, self.img_size_y, rs.format.z16, 15)

        profile = self.pipeline.start(self.config)

        # 화이트밸런스 수동 설정 추가 부분
        device = profile.get_device()
        color_sensor = device.query_sensors()[1]  # 보통 1이 컬러 센서임
        # 자동 화이트밸런스는 끄지 않고, 수동 값만 설정
        color_sensor.set_option(rs.option.white_balance, 5800)  # 원하는 값으로 설정 (2800~6500 권장)

        self.align = rs.align(rs.stream.color)

        # 추가 필터
        self.hole_filling_filter = rs.hole_filling_filter()
        self.temporal_filter = rs.temporal_filter()
        self.spatial_filter = rs.spatial_filter()

        self.capture_timer = self.create_timer(1/15, self.image_capture)
        self.process_timer = self.create_timer(1/15, self.image_processing)

        self.center_x = int(self.img_size_x / 2)
        self.color_img = np.zeros((self.img_size_y, self.img_size_x, 3), dtype=np.uint8)
        self.depth_img = np.zeros((self.img_size_y, self.img_size_x), dtype=np.uint16)

        self.L_sum = 0
        self.R_sum = 0

    def image_capture(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # 필터 적용
        depth_frame = self.spatial_filter.process(depth_frame)
        depth_frame = self.temporal_filter.process(depth_frame)
        depth_frame = self.hole_filling_filter.process(depth_frame)

        self.color_img = np.asanyarray(color_frame.get_data())
        self.depth_img = np.asanyarray(depth_frame.get_data())

    def yuv_detection(self, img):
        y, x, c = img.shape

        gaussian = cv2.GaussianBlur(img, (3, 3), 1)
        yuv_img = cv2.cvtColor(gaussian, cv2.COLOR_BGR2YUV)
        Y_img, U_img, V_img = cv2.split(yuv_img)

        ret, U_img_treated = cv2.threshold(U_img, self.U_detection_threshold, 255, cv2.THRESH_BINARY)

        max_contour = None
        filtered = np.zeros_like(img)

        if ret:
            contours, _ = cv2.findContours(U_img_treated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            max_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_contour = contour

            if max_contour is not None:
                max_contour_mask = np.zeros_like(U_img_treated)
                cv2.drawContours(max_contour_mask, [max_contour], -1, 255, thickness=cv2.FILLED)

                filtered = cv2.bitwise_and(img, img, mask=max_contour_mask)
                cv2.imshow("Filtered U Channel", filtered)
                cv2.waitKey(1)

        return max_contour, filtered

    def image_processing(self):
        x1 = int(self.img_size_x * 0.05)
        x2 = int(self.img_size_x * 0.95)
        y1 = int(self.img_size_y * 0.7)
        y2 = int(self.img_size_y * 0.9)

        roi = self.color_img[y1:y2, x1:x2]
        roi_depth = self.depth_img[y1:y2, x1:x2]

        max_contour, filtered = self.yuv_detection(roi)

        Lsum = 0
        Rsum = 0

        if max_contour is not None:
            mask = np.zeros_like(filtered[:, :, 0])
            cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)

            # 깊이 마스킹 (0.3m ~ 2.5m)
            depth_mask = np.where((roi_depth > 300) & (roi_depth < 2500), 255, 0).astype(np.uint8)
            combined_mask = cv2.bitwise_and(mask, depth_mask)

            width = mask.shape[1]
            mid = width // 2
            Lsum = cv2.countNonZero(combined_mask[:, :mid])
            Rsum = cv2.countNonZero(combined_mask[:, mid:])

        self.L_sum = Lsum
        self.R_sum = Rsum

        left, center, right = None, None, None
        if max_contour is not None:
            x, y, w, h = cv2.boundingRect(max_contour)
            left = x
            right = x + w
            center = int((left + right) / 2)

        if center is not None:
            alpha = 0.2
            target_center_x = x1 + center
            self.center_x = int((1 - alpha) * self.center_x + alpha * target_center_x)

            roi_center_x = (x2 - x1) // 2
            angle_offset = center - roi_center_x
            self.get_logger().info(f"Angle Offset (center deviation): {angle_offset}")
            self.get_logger().info(f"Lsum: {self.L_sum}, Rsum: {self.R_sum}")

            cv2.line(self.color_img, (self.center_x, y1), (self.center_x, y2), (255, 0, 0), 4)
            cv2.line(roi, (roi_center_x, 0), (roi_center_x, y2 - y1), (0, 255, 0), 2)
            cv2.line(roi, (left, 0), (left, y2 - y1), (0, 255, 255), 2)
            cv2.line(roi, (right, 0), (right, y2 - y1), (0, 255, 255), 2)

            cv2.drawContours(roi, [max_contour], -1, (0, 0, 255), 2)

        cv2.rectangle(self.color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(self.color_img, f'L : {self.L_sum}   R : {self.R_sum}', (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        cv2.imshow("Color Image", self.color_img)
        cv2.imshow("ROI", roi)
        cv2.imshow("Filtered U + Depth Masked", filtered)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = Lanedecttest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
