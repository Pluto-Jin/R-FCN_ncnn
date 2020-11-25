// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"
#include "layer/proposal.h"
#include <iostream>

#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>

struct Object
{
    cv::Rect_<float> rect; //x1,y1,w,h
    float pos[4]; //x1,y1,x2,y2
    int label; 
    float prob;
};

void print_mat(const ncnn::Mat& m) 
{
    for (int q=0;q<5 and q<m.c;q++)
    {
        const float* p=m.channel(q);
        for (int y=0,cnt=10;cnt and y<m.h;y++) {
            for (int x=0;cnt and x<m.w;x++) fprintf(stderr,"%.4f ",p[x]),cnt--;
            p+=m.w;
        }
        fprintf(stderr,"\n");
    } 
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void box_decode(float* pb,float dx,float dy,float dw,float dh,float sx,float sy,float ex,float ey) {

    const float ma = 4.135166556742356;

    // shifted anchor
    float anchor_w = ex - sx + 1;
    float anchor_h = ey - sy + 1;
    float anchor_x = sx;
    float anchor_y = sy;


    float cx = anchor_x + anchor_w * 0.5f;
    float cy = anchor_y + anchor_h * 0.5f;
    
    // apply center size
    dx/=10.,dy/=10.,dw/=5.,dh/=5.;
    dw=std::min(dw,ma);
    dh=std::min(dh,ma);

    float pb_cx = cx + anchor_w * dx;
    float pb_cy = cy + anchor_h * dy;

    float pb_w = anchor_w * exp(dw);
    float pb_h = anchor_h * exp(dh);
    //if (pb_w>200 and pb_h>200) ffprintf(stderr,"%.4f %.4f\n",dw,dh);


    pb[0] = pb_cx - pb_w * 0.5f;
    pb[1] = pb_cy - pb_h * 0.5f;
    pb[2] = pb_cx + pb_w * 0.5f - 1;
    pb[3] = pb_cy + pb_h * 0.5f - 1;
}

static float bilinear_interpolate(const float *bottom_data, const int height, const int width, float y, float x) {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
        return 0;
    }

    if (y <= 0) y = 0;
    if (x <= 0) x = 0;

    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (float)y_low;
        } else {
            y_high = y_low + 1;
        }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (float)x_low;
        } else {
            x_high = x_low + 1;
        }

    float ly = y - y_low;
    float lx = x - x_low;
    float hy = 1. - ly;
    float hx = 1. - lx;
    // do bilinear interpolation
    float lt = bottom_data[y_low * width + x_low];
    float rt = bottom_data[y_low * width + x_high];
    float lb = bottom_data[y_high * width + x_low];
    float rb = bottom_data[y_high * width + x_high];
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    float val = (w1 * lt + w2 * rt + w3 * lb + w4 * rb);

    return val;
}

static float ps_roi_align_grid(const float* fmap,int h,int w,float sy,float sx,int col,int row,float bh,float bw) {
    float ma=-1e20;
    for (int iy=0;iy<2;iy++) {
        float y=sy+bh*(1.*row+0.25+0.5*iy);
        for (int ix=0;ix<2;ix++) {
            float x=sx+bw*(1.*col+0.25+0.5*ix);
            float cur=bilinear_interpolate(fmap,h,w,y,x);
            if (cur>ma) ma=cur;
        }
    }
    return ma;
}

static int detect_rfcn(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net rfcn;
    ncnn::Proposal pro_layer;

    rfcn.opt.use_vulkan_compute = false;

    // original pretrained model from https://github.com/YuwenXiong/py-R-FCN
    // https://github.com/YuwenXiong/py-R-FCN/blob/master/models/pascal_voc/ResNet-50/rfcn_end2end/test_agnostic.prototxt
    // https://1drv.ms/u/s!AoN7vygOjLIQqUWHpY67oaC7mopf
    // resnet50_rfcn_final.caffemodel
    //rfcn.load_param("rfcn_end2end.param");
    //rfcn.load_model("rfcn_end2end.bin");
    rfcn.load_param("rfcn.param");
    rfcn.load_model("rfcn.bin");

    const int target_size = 224;

    const int max_per_image = 20;
    const float confidence_thresh = 0.0001f; // CONF_THRESH

    const float nms_threshold = 0.5f; // NMS_THRESH

    // scale to target detect size
    int w = bgr.cols;
    int h = bgr.rows;
    float scale = 1.f;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_RGB2BGR, bgr.cols, bgr.rows, w, h);

    const float mean_vals[3] = {255.0*0.485f, 255.0*0.456f, 255.0*0.406f};
    const float norm_vals[3] = {1.0/255.0/0.229f, 1.0/255.0/0.224f, 1.0/255.0/0.225f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    //print_mat(in);

    ncnn::Mat im_info(3);
    im_info[0] = h;
    im_info[1] = w;
    im_info[2] = scale;

    //fprintf(stderr,"image_shape: %d*%d\n",w,h);
    // step1, extract feature and all rois
    ncnn::Extractor ex1 = rfcn.create_extractor();

    ex1.input("input", in);
    //ex1.input("data", in);
    //ex1.input("im_info", im_info);

    ncnn::Mat rfcn_cls,rfcn_bbox,psf,psf4;
    /*
       ex1.extract("rfcn_cls", rfcn_cls);
       ex1.extract("rfcn_bbox", rfcn_bbox);
       ex1.extract("rois", rois);
     */
    ex1.extract("score", rfcn_cls);
    ex1.extract("reg", rfcn_bbox);
    ex1.extract("psf", psf);
    ex1.extract("psf4", psf4);
    //print_mat(feature);

    //fprintf(stderr, "cls shape: %d %d %d\n", rfcn_cls.c, rfcn_cls.h, rfcn_cls.w);
    //fprintf(stderr, "bbox shape: %d %d %d\n", rfcn_bbox.c, rfcn_bbox.h, rfcn_bbox.w);
    //fprintf(stderr, "psf shape: %d %d %d\n", psf.c, psf.h, psf.w);
    //fprintf(stderr, "psf4 shape: %d %d %d\n", psf4.c, psf4.h, psf4.w);
    //print_mat(rfcn_bbox);
    //print_mat(rfcn_cls);
    //print_mat(psf);
    //print_mat(psf4);

    std::vector<ncnn::Mat> top(2),bot(3);
    ncnn::Mat& rois=top[0]; // all rois
    bot[0]=rfcn_cls, bot[1]=rfcn_bbox, bot[2]=im_info;
    //print_mat(bot[0]);
    //print_mat(bot[1]);

    pro_layer.load_param();
    pro_layer.forward(bot,top,rfcn.opt);

    //fprintf(stderr, "rois: %d %d %d\n", rois.c, rois.h, rois.w);
    //fprintf(stderr, "scores: %d %d %d\n", top[1].c, top[1].h, top[1].w);
    //print_mat(rois);
    //print_mat(top[1]);

    // step2, extract bbox and score for each roi
    int height=psf.h,width=psf.w;
    std::vector<std::vector<Object> > class_candidates(4);
    for (int i = 0; i < rois.c; i++) {
        const float* roi = rois.channel(i); // get single roi
        float sx=roi[0]/32,sy=roi[1]/32,ex=roi[2]/32,ey=roi[3]/32;
        float h=ey-sy,w=ex-sx,bh=h/7,bw=w/7;
        std::vector<float> scores(4);
        for (int cl=0;cl<4;cl++) {
            float result=0;
            for (int ind=0;ind<49;ind++) {
                int row=ind/7,col=ind%7;
                const float* fmap=psf.channel(cl*49+ind);
                result+=ps_roi_align_grid(fmap,height,width,sy,sx,col,row,bh,bw);
            }
            scores[cl]=result/49;
        }
        float den=exp(scores[0])+exp(scores[1])+exp(scores[2])+exp(scores[3]), ma=0;
        //int cls=-1;
        for (int j=0;j<4;j++) {
            scores[j]=exp(scores[j])/den;
            //if (scores[j]>ma) ma=scores[j],cls=j;
        }

        for (int cl=0;cl<4;cl++) {
            float dx=0,dy=0,dw=0,dh=0;
            for (int ind=0;ind<49;ind++) {
                int row=ind/7,col=ind%7;
                dx+=ps_roi_align_grid(psf4.channel(cl*196+ind),height,width,sy,sx,col,row,bh,bw);
                dy+=ps_roi_align_grid(psf4.channel(cl*196+ind+49*1),height,width,sy,sx,col,row,bh,bw);
                dw+=ps_roi_align_grid(psf4.channel(cl*196+ind+49*2),height,width,sy,sx,col,row,bh,bw);
                dh+=ps_roi_align_grid(psf4.channel(cl*196+ind+49*3),height,width,sy,sx,col,row,bh,bw);
                //fprintf(stderr,"%.4f %.4f %.4f %.4f\n",dx,dy,dw,dh);
            }
            float tmp[4];
            box_decode(tmp,dx/49,dy/49,dw/49,dh/49,roi[0],roi[1],roi[2],roi[3]);
            //fprintf(stderr,"%.4f %.4f %.4f %.4f\n",dx/49,dy/49,dw/49,dh/49);
            //fprintf(stderr,"%.4f %.4f %.4f %.4f %d %.4f\n",roi[0],roi[1],roi[2],roi[3],cl,ma);

            Object obj;
            obj.label = cl;
            obj.prob = scores[cl];

            // ignore background or low score
            if (cl == 0 || scores[cl] <= confidence_thresh)
                continue;

            // clip
            tmp[0] = std::max(std::min(tmp[0], (float)(bgr.cols - 1)), 0.f);
            tmp[1] = std::max(std::min(tmp[1], (float)(bgr.rows - 1)), 0.f);
            tmp[2] = std::max(std::min(tmp[2], (float)(bgr.cols - 1)), 0.f);
            tmp[3] = std::max(std::min(tmp[3], (float)(bgr.rows - 1)), 0.f);

            for (int i=0;i<4;i++) obj.pos[i]=tmp[i];
            obj.rect = cv::Rect_<float>(tmp[0], tmp[1], tmp[2]-tmp[0]+1, tmp[3]-tmp[1]+1);
            //fprintf(stderr,"%.4f %.4f %.4f %.4f %d %.4f\n",tmp[0],tmp[1],tmp[2],tmp[3],cl,ma);
            class_candidates[cl].push_back(obj);
        }

    }

    // post process
    objects.clear();
    for (int i = 0; i < (int)class_candidates.size(); i++)
    {
        std::vector<Object>& candidates = class_candidates[i];

        qsort_descent_inplace(candidates);

        std::vector<int> picked;
        nms_sorted_bboxes(candidates, picked, nms_threshold);

        for (int j = 0; j < (int)picked.size(); j++)
        {
            int z = picked[j];
            objects.push_back(candidates[z]);
        }
    }

    qsort_descent_inplace(objects);

    if (max_per_image > 0 && max_per_image < objects.size())
    {
        objects.resize(max_per_image);
    }
    
    int cnt=0;
    for (int i=0;i<objects.size();i++) {
        auto obj=objects[i];
        if (obj.prob>0.3) cnt++;
        //fprintf(stderr, "%d = %.5f at %.4f %.4f %.4f %.4f\n", obj.label, obj.prob, obj.pos[0], obj.pos[1], obj.pos[2], obj.pos[3]);
    }
    objects.resize(cnt);

    //for (auto obj:objects) fprintf(stderr, "final: %d = %.5f at %.4f %.4f %.4f %.4f\n", obj.label, obj.prob, obj.pos[0], obj.pos[1], obj.pos[2], obj.pos[3]);
    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"background", "bus",
        "car", "van", "others"  };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_rfcn(m, objects);

    draw_objects(m, objects);

    return 0;
}
