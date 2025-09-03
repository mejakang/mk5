import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, Joy
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import math

class BlueRatioCirculator(Node):
    def __init__(self):
        super().__init__('testmove')

        qos_profile = QoSProfile(depth=10)
        img_qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1) #이미지 실시간성

        self.control_publisher = self.create_publisher(Float32MultiArray, 'Odrive_control', qos_profile)
        self.img_publisher = self.create_publisher(Image, 'img_data', img_qos_profile)
        self.center_publisher = self.create_publisher(Float32MultiArray, 'center_x', qos_profile)

        self.joy_subscriber = self.create_subscription(Joy, 'joy', self.joy_msg_sampling, qos_profile)
        self.imu_subscriber = self.create_subscription(Float32MultiArray, 'imu_data', self.imu_msg_sampling, qos_profile)
        self.encoder_subscriber = self.create_subscription(Float32MultiArray, 'Odrive_encoder', self.encoder_clear, qos_profile)

        self.capture_timer = self.create_timer(1/15, self.image_capture)
        self.process_timer = self.create_timer(1/15, self.image_processing)
        self.pub_control = self.create_timer(1/15, self.track_tracking)

        self.U_detection_threshold = 130
        self.img_size_x = 848
        self.img_size_y = 480
        self.depth_size_x = 848
        self.depth_size_y = 480
        self.max_speed = 10
        self.odrive_mode = 1.
        self.joy_status = False
        self.joy_stick_data = [0, 0]

        self.center_x = int(self.img_size_x / 2)  # 파란색 중심선 x좌표 초기화
        self.chess_detection_flag = False

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.img_size_x, self.img_size_y, rs.format.bgr8, 15)
        self.config.enable_stream(rs.stream.depth, self.depth_size_x, self.depth_size_y, rs.format.z16, 15)
        profile = self.pipeline.start(self.config)
        self.depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.hole_filling_filter = rs.hole_filling_filter()
        self.temporal_filter = rs.temporal_filter()
        self.spatial_filter = rs.spatial_filter()

        self.cvbridge = CvBridge()
        self.chess_model = YOLO('/home/ljh/goodbox-project/train_goodbox_highacc/weights/chess.pt')  # 체스판
        self.goodbox_model = YOLO('/home/ljh/goodbox-project/train_goodbox_highacc/weights/goodbox.pt')  # 보급

        self.finish_ROI = [
            [int(self.img_size_x * 0.45), int(self.img_size_y * 0.6)],
            [int(self.img_size_x * 0.55), int(self.img_size_y * 0.7)]
        ]

        # 화이트밸런스 수동 설정 추가 부분
        device = profile.get_device()
        color_sensor = device.query_sensors()[1]  # 보통 1번이 컬러 센서
        color_sensor.set_option(rs.option.enable_auto_white_balance, 1)  # 자동 켜기(0이면 끄기)

    def encoder_clear(self, msg):
        self.encoder = msg.data
        self.get_logger().info(f"Encoder L={self.encoder[0]:.2f}, R={self.encoder[1]:.2f}")

    def image_capture(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        self.color_frame = aligned_frames.get_color_frame()
        self.aligned_depth_frame = aligned_frames.get_depth_frame()
        depth_filtered = self.temporal_filter.process(self.aligned_depth_frame)
        depth_filtered = self.spatial_filter.process(depth_filtered)
        self.filled_depth_frame = self.hole_filling_filter.process(depth_filtered)
        self.depth_intrinsics = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        self.depth_img = np.asanyarray(self.filled_depth_frame.get_data())
        self.color_img = np.asanyarray(self.color_frame.get_data())

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

            # self.center_x를 ROI 기준에서 전체 이미지 기준으로 변환
            # yuv_detection에 들어오는 img는 ROI (x1_roi ~ x2_roi) 범위이므로 offset 필요
            roi_x_offset = int(self.img_size_x * 0.05)  # ROI 시작 x 좌표
            self.center_x = roi_x_offset + center

            # 파란색 중심선 그리기 (원래 컬러 이미지에)
            cv2.line(self.color_img, (self.center_x, int(self.img_size_y * 0.7)), (self.center_x, int(self.img_size_y * 0.9)), (255, 0, 0), 2)

            # 경계 라인 좌우 표시
            cv2.line(img, (left, 0), (left, y), (0, 255, 255), 1)
            cv2.line(img, (right, 0), (right, y), (0, 255, 255), 1)
            cv2.drawContours(img, [max_contour], -1, (0, 0, 255), 2)

            #self.get_logger().info(f"Blue line center x: {self.center_x}")

            try:
                self.img_publisher.publish(self.cvbridge.cv2_to_imgmsg(filtered, encoding='bgr8'))
            except Exception as e:
                self.get_logger().error(f"Image publishing failed: {e}")

        return

    def image_processing(self):
        if not hasattr(self, 'color_img') or self.color_img is None:
            return

        # ROI 좌표
        x1_roi = int(self.img_size_x * 0.05)
        y1_roi = int(self.img_size_y * 0.7)
        x2_roi = int(self.img_size_x * 0.95)
        y2_roi = int(self.img_size_y * 0.9)

        # ROI 지정
        roi = self.color_img[y1_roi:y2_roi, x1_roi:x2_roi]

        # 파란색 경계 중심 추출
        self.yuv_detection(roi)

        # 체스판 검출
        results = self.chess_model.predict(self.color_img, conf=0.6, verbose=False, max_det=1)
        if results and results[0].boxes.xywh.numel() > 0:
            box = results[0].boxes.xywh[0].detach().cpu().numpy().astype(int)
            x, y, w, h = box
            center_x, center_y = x, y
            if (self.finish_ROI[0][0] < center_x < self.finish_ROI[1][0] and
                self.finish_ROI[0][1] < center_y < self.finish_ROI[1][1]):
                self.chess_detection_flag = True
                self.get_logger().info("Finish line detected!")

            x1_box, y1_box = x - w//2, y - h//2
            x2_box, y2_box = x + w//2, y + h//2
            cv2.rectangle(self.color_img, (x1_box, y1_box), (x2_box, y2_box), (0, 0, 255), 2)

        # 굿박스 검출 및 3D 좌표 추출
        goodbox_results = self.goodbox_model.predict(self.color_img, conf=0.6, verbose=False)
        for box in goodbox_results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)
            depth = self.aligned_depth_frame.get_distance(u, v)
            point_3d = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [u, v], depth)
            X, Y, Z = point_3d
            self.get_logger().info(f"Detected goodbox at (u,v)=({u},{v}), 3D=({X:.2f}, {Y:.2f}, {Z:.2f})")
            cv2.rectangle(self.color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.color_img, "goodbox", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ROI 시각화
        cv2.rectangle(self.color_img, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 0), 2)

        cv2.imshow("Color", self.color_img)
        cv2.imshow("ROI", roi)
        cv2.waitKey(1)

    def track_tracking(self):
        msg = Float32MultiArray()

        if self.chess_detection_flag:
            self.L_joy = 0.0
            self.R_joy = 0.0
            self.get_logger().info("체스판 감지 - 정지")
        elif self.joy_status:
            self.L_joy = self.joy_stick_data[0] * self.max_speed
            self.R_joy = self.joy_stick_data[1] * self.max_speed
            self.get_logger().info("수동 입력 모드")
        else:
            # 자동 추적: 파란색 중심선을 따라 이동
            center = self.center_x
            center_image = int(self.img_size_x / 2)
            error = center_image - center

            # 비례 제어 (P-controller) 기본
            Kp = 0.01
            control = Kp * error

            base_speed = self.max_speed * 0.5
            left_speed = base_speed + control
            right_speed = base_speed - control

            # 속도 한계 clamp
            self.L_joy = max(min(left_speed, self.max_speed), -self.max_speed)
            self.R_joy = max(min(right_speed, self.max_speed), -self.max_speed)

            self.get_logger().info(f"자동 추적: center={center}")
            #self.get_logger().info(f"자동 추적: center={center}, error={error}, L_joy={self.L_joy:.2f}, R_joy={self.R_joy:.2f}")

        msg.data = [self.odrive_mode, self.L_joy, self.R_joy]
        self.control_publisher.publish(msg)

    def joy_msg_sampling(self, msg):
        axes = msg.axes
        self.joy_status = axes[2] != 1
        if self.joy_status:
            self.joy_stick_data = [axes[1], axes[4]]

    def imu_msg_sampling(self, msg):
        if msg.data[0] <= 77.5:
            self.robot_roll = -1
        elif msg.data[0] >= 105:
            self.robot_roll = 1
        else:
            self.robot_roll = 0

        self.theta = msg.data[1]
        self.get_logger().info(f"IMU Theta: {self.theta:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = BlueRatioCirculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
