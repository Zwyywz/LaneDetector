//
// Created by zhouwy on 2022/3/31.
//
#include <iostream>
#include  <numeric>
#include <opencv2/opencv.hpp>

cv::Mat doCanny(const cv::Mat &image);

cv::Mat doSegment(const cv::Mat &image);

cv::Mat polyfit(std::vector<cv::Point> &in_point, int n);

cv::Mat visualize_lines(const cv::Mat &frame, std::vector<double> coordinates);

std::vector<double> calculateLines(const cv::Mat &image, std::vector<cv::Vec4i> lines);

std::vector<double> calculate_coordinates(cv::Mat image, double SlopeAvg, double YInterceptAvg);

cv::Mat doCanny(const cv::Mat &image) {
    /* @function: 对输入图片做边缘检测，得到目标图片中突出的边缘线
     * @param: image -> 输入图片
     * @return： canny -> canny 边缘检测后的二值图
     * */
    cv::Mat gray, gauss, canny;
    cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
    cv::GaussianBlur(gray, gauss, cv::Size(5, 5), 0);
    cv::Canny(gauss, canny, 50, 150);
    return canny;
}

cv::Mat doSegment(const cv::Mat &image) {
    /* @function: 分割出车道线的区域
     * @param: image -> 输入二值图
     * @return： 按points(3)中三点分割出三角形区域
     * */
    int height = image.rows;
    cv::Mat segment;
    std::vector<cv::Point> points(3);
    points[0] = cv::Point(0, height);
    points[1] = cv::Point(800, height);
    points[2] = cv::Point(380, 290);
    cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    cv::fillPoly(mask, points, 255);
    cv::bitwise_and(image, mask, segment);
    return segment;
}

cv::Mat polyfit(std::vector<cv::Point> &in_point, int n) {
    int size = in_point.size();
    //所求未知数个数
    int x_num = n + 1;
    //构造矩阵U和Y
    cv::Mat mat_u(size, x_num, CV_64F);
    cv::Mat mat_y(size, 1, CV_64F);

    for (int i = 0; i < mat_u.rows; ++i)
        for (int j = 0; j < mat_u.cols; ++j) {
            mat_u.at<double>(i, j) = pow(in_point[i].x, j);
        }

    for (int i = 0; i < mat_y.rows; ++i) {
        mat_y.at<double>(i, 0) = in_point[i].y;
    }
    //矩阵运算，获得系数矩阵K
    cv::Mat mat_k(x_num, 1, CV_64F);
    mat_k = (mat_u.t() * mat_u).inv() * mat_u.t() * mat_y;
    //std::cout << "参数：" << mat_k << std::endl;
    return mat_k;
}

std::vector<double> calculateLines(const cv::Mat &image, std::vector<cv::Vec4i> lines) {
    /* @function: 将霍夫直线检测后得到的坐标进行处理，先做多项式拟合得到斜率和截距，再求斜率和截距平均值。
     * @param: image -> 输入图片
     * @param: lines -> 霍夫直线检测后得到的坐标
     * @return: 拟合后的左右两边的起始和终点坐标。
     *           std::vector<double> result = [x_left_start, y_left_start, x_left_end, y_left_end,x_right_start, y_right_start, x_right_end, y_right_end]
     * */
    std::vector<double> leftSlope, rightSlope, leftYIntercept, rightYIntercept;
    for (std::vector<cv::Vec4i>::iterator it = lines.begin(); it < lines.end(); it++) {
        cv::Point2i startXY((*it)[0], (*it)[1]);
        cv::Point2i endXY((*it)[2], (*it)[3]);
        cv::Point in[2] = {startXY, endXY};
        std::vector<cv::Point> in_point(std::begin(in), std::end(in));
        cv::Mat parameters = polyfit(in_point, 1);
        double y_intercept = parameters.at<double>(0);
        double slope = parameters.at<double>(1);
        if (slope < 0) {
            leftSlope.emplace(leftSlope.end(), slope);
            leftYIntercept.emplace(leftYIntercept.end(), y_intercept);
        } else {
            rightSlope.emplace(rightSlope.end(), slope);
            rightYIntercept.emplace(rightYIntercept.end(), y_intercept);
        }
    }

    double leftSlopeAvg =
            std::accumulate(std::begin(leftSlope), std::end(leftSlope), double(0.0)) / double(leftSlope.size());
    double rightSlopeAvg =
            std::accumulate(std::begin(rightSlope), std::end(rightSlope), double(0.0)) / double(rightSlope.size());
    double leftYInterceptAvg =
            std::accumulate(std::begin(leftYIntercept), std::end(leftYIntercept), double(0.0)) /
            double(leftYIntercept.size());
    double rightYInterceptAvg =
            std::accumulate(std::begin(rightYIntercept), std::end(rightYIntercept), double(0.0)) /
            double(rightYIntercept.size());

    std::vector<double> left_line = calculate_coordinates(image, leftSlopeAvg, leftYInterceptAvg);
    std::vector<double> right_line = calculate_coordinates(image, rightSlopeAvg, rightYInterceptAvg);
    std::vector<double> result;
    result.resize(left_line.size() + right_line.size());
    std::merge(left_line.begin(), left_line.end(), right_line.begin(), right_line.end(), result.begin());
    std::cout << "calculateLines: ";
    for (std::vector<double>::iterator it = result.begin(); it < result.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    return result;
}

std::vector<double> calculate_coordinates(cv::Mat image, double SlopeAvg, double YInterceptAvg) {
    std::vector<double> result;
    int y1 = image.rows;
    int y2 = int(y1 - 150);
    //Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    int x1 = int((y1 - YInterceptAvg) / SlopeAvg);
    //Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    int x2 = int((y2 - YInterceptAvg) / SlopeAvg);
    result.emplace(std::end(result), x1);
    result.emplace(std::end(result), y1);
    result.emplace(std::end(result), x2);
    result.emplace(std::end(result), y2);
    return result;
}

cv::Mat visualize_lines(const cv::Mat &frame, std::vector<double> coordinates) {
    /* @function: 将直线画在与frame大小相同的mask上
     * @param: frame -> 输入图片
     * @param: coordinates -> 坐标：
     *                std::vector<double> coordinates = [x_left_start, y_left_start, x_left_end, y_left_end,x_right_start, y_right_start, x_right_end, y_right_end]
     * @return: 画上直线的掩码mask
     * */
    cv::Mat mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
    for (unsigned long i = 0; i < coordinates.size(); i += 4) {
        if (i < 4)
            cv::line(mask, cv::Point2d(coordinates[i], coordinates[i + 1]),
                     cv::Point2d(coordinates[i + 2], coordinates[i + 3]), cv::Scalar(0, 255, 0), 3);
        else
            cv::line(mask, cv::Point2d(coordinates[i], coordinates[i + 1]),
                     cv::Point2d(coordinates[i + 2], coordinates[i + 3]), cv::Scalar(255, 0, 0), 3);
    }
    return mask;
}


int main(int argc, char *argv[]) {
    cv::Mat image, canny, segment, visualize, output;
    std::vector<cv::Vec4i> lines;
    std::vector<double> coordinates;
    std::cout << "OpenCV Version:" << CV_VERSION << std::endl;
    cv::VideoCapture cap("../input.mp4");

    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    } else {
        std::cout << "success" << std::endl;
    }
    cv::namedWindow("input");
    cv::namedWindow("canny");
    cv::namedWindow("Segment");
    //cv::namedWindow("Hough");
    cv::namedWindow("output");



    while (true) {
        if (!cap.read(image)) {
            std::cout << "No frame" << std::endl;
            cv::waitKey();
        }
        //边缘检测
        canny = doCanny(image);
        //分割
        segment = doSegment((canny));
        //霍夫直线检测
        cv::HoughLinesP(segment, lines, 2, CV_PI / 180, 100, 100, 50);
        //cv::line(image, cv::Point2d(lines[0][0], lines[0][1]), cv::Point2d(lines[0][2], lines[0][3]),
        //         cv::Scalar(0, 255, 0), 2);
        //cv::imshow("Hough", image);
        //计算坐标
        coordinates = calculateLines(image, lines);
        visualize = visualize_lines(image, coordinates);
        cv::addWeighted(image, 0.9, visualize, 1, 1, output);


        cv::imshow("input", image);
        cv::imshow("canny", canny);
        cv::imshow("Segment", segment);
        cv::imshow("output", output);


        if (cv::waitKey(1) >= 0) break;
    }
}
