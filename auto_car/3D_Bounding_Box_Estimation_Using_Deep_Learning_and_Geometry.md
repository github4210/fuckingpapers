----
title: 3D Bounding Box Estimation Using Deep Learning and Geometry
date: 2017-06-29 23:21:21
category: 默认分类
---

本文介绍 3D Bounding Box Estimation Using Deep Learning and Geometry
<!-- more -->
# 3D Bounding Box Estimation Using Deep Learning and Geometry
> This article was original written by Jin Tian, welcome re-post, but please keep this copyright info, thanks, any question could be asked via wechat: `jintianiloveu` 

# 摘要
这篇文章说白了就是通过一张单一的图片来预测object的pose，以及立体位置。由于最近想做一个类似的demo，因此阅读一下这个论文，看上去是一个大婶在公司实习的时候完成的工作，牛逼到爆炸。文中使用了两个网络，其中一个是用`hybrid discrete-continuous loss`，这啥？网格离散连续loss？不知道什么鬼。另外一个网络是回归3D物体的dimensions，其实就是目标检测里面预测bbox。文中提到使用这个几何方法秒杀其他计算量巨大的算法，包括一些基于分割的方法，总的来说速度快精度高非常牛逼。

