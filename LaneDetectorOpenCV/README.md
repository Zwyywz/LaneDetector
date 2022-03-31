# 基于OpenCV和C++车道线检测





前言： 这篇文章主要是一种基于边缘检测的传统车道线检测算法，这种算法完全依赖于OpenCV，兼顾了轻量级和实时性。代码开源：https://github.com/Zwyywz/laneDetector

![inputVideo](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/inputVideo.gif)


## 1、canny边缘检测

**Canny边缘检测算法主要步骤：高斯滤波、梯度计算、非极大值抑制和双阈值检测。**

- 高斯滤波：使用高斯滤波的目的是平滑图像，滤除图像中的部分噪声（因为微分算子对噪声很敏感）
- 梯度计算：图像也可以计算梯度，由于数字图像是有离散的像素点的灰度值构成，所以微分运算就变成了差分，我们可以用相邻两个像素点之间的差分值表示该像素点在某个方向上灰度的变化情况。
- 非极大值抑制：细化边缘，梯度计算得到的边缘很粗，一条边缘中央一般很亮，两边亮度逐渐降低，可以根据这个特点去掉非局部灰度最高的“假边”，达到细化边缘的目的。
- 双阈值检测：减少伪边缘点非极大值抑制之后，检测到的边缘线条比较多，我们可以滤掉一些很暗的边缘，并让主要的边缘凸显出来。

**OpenCV中提供了Canny边缘检测的方法，只需要调用对应的函数即可**

```c++
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

```

Canny边缘检测后的结果如下：

![canny](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/canny.gif)

## 2、图片分割

图片分割的主要思想是，构建一个mask，这个mask包含主要的车道区域值均为1，其余为0，将这个mask与原frame进行叠加，就可以抠出主要的车道区域。

可以观察到：待识别的车道线主要分布于一个三角形区域内，我们可以估计这个三角形区域大概的顶点坐标，从而得到mask。

```c++
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

```

![segment](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/segment.gif)



## 3、hough直线检测

首先，介绍笛卡尔空间，就是我们常见的那个几何空间，通过 y=kx+b，可以表示直线。

然后，想一下，如果把上面方程变形一下，b=-xk+y，（k和b作为变量，x和y作为常量），那么是不是又是一条另外的直线呢？对了，这就是霍夫空间了。

- 霍夫空间，笛卡尔空间中的直线，对应到霍夫空间中是一个点。
- 笛卡尔空间中共线的点，在霍夫空间中对应的直线相交。这个很重要，因为在笛卡尔空间中，我要做直线检测，岂不就是要找到最多的点所在的那条线嘛，我把所有的点都映射到霍夫空间中，找到最多的线公共交点就可以了。

再然后，会发现一个问题，如果是一条垂直于x轴的直线，那么k岂不是正无穷，怎么办？

没关系，引进极坐标表示直线：ρ=xCosθ+ySinθ（ρ为原点到直线的距离）,如图所示

![](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/image-20220327160437866.png)



再然后，就可以将笛卡尔空间和霍夫空间做映射了。

![](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/20180329185654863.png)

继续然后，上面红色字体：找到最多的线公共交点就可以了，怎么找？

直接小白方式寻找，把霍夫空间网格化（就是一个很大的矩阵，初始值全是0），直线经过的地方标注1，没经过的地方还是0。再找出值最大的一些点就可以了。

**OpenCV中提供了Hough直线检测的方法，只需要调用对应的函数即可**

```c++
cv::HoughLinesP(segment, lines, 2, CV_PI / 180, 100, 100, 50);
cv::line(image,cv::Point2d(lines[0][0],lines[0][1]),cv::Point2d(lines[0][2],lines[0][3]),cv::Scalar(0,255,0),2);
cv::imshow("Hough",image);
```

![Hough](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/Hough.gif)

不难观察出，由霍夫直线检测所得到的坐标点太多，导致检测出的直线左右闪动，所以我们要想办法将所有的坐标点变化量减小，所以平滑滤波？？？好吧！求个平均值就行！

## 4、坐标计算

将从hough检测到的多条线平均成一条线表示车道的左边界， 一条线表示车道的右边界。基本思想很简单，**就是先将霍夫变换的线段转换为一维信息，进行多项式拟合，在将得到的截距和斜率信息进行平均，在利用数值代换转换成cv坐标系的左边界线，和右边界线；**

首先，先实现一个多项式拟合函数，基本原理：**幂函数可逼近任意函数。**
$$
y = \sum_{i = 0}^{n}kx^i
$$

```c++
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
```

我们只需要得到斜率和截距，故而只需要拟合成一次多项式。

接下来正式处理hough直线检测后的坐标点：

```c++
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
//    std::cout << "calculateLines: ";
//    for (std::vector<double>::iterator it = result.begin(); it < result.end(); ++it) {
//        std::cout << *it << " ";
//    }
//    std::cout << std::endl;
    return result;
}
```

结果如下：

![output](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/output.gif)

## 5、结果可视化

最后，将得到的坐标通过OpenCV中画图工具即可叠加到原图上：

```c++

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
```

![result](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/result.gif)

