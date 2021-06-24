How to analyze results obtained with the GazeClassify package on R ?
================

First of all, we need to import our csv file and manipulate the data to
get the results we need. The final “Result” is a dataframe containing
for every joint the pourcentage of frames where it was the closest to
gaze irrelevantly of wether the gaze is located on or outside of the
human shape.

``` r
Data <- read.table("C:/Users/flori/OneDrive/Bureau/Internship/Result.csv",header = TRUE, sep=",",dec=".")
Shape <- Data %>% filter(name == 'Human_Shape')
Shape_frame <- Shape[order(Shape$frame),]
Joint <- Data %>% filter(name == 'Human_Joints') %>% group_by(frame) %>% filter(distance == min(distance))
Joint_frame <- Joint[order(Joint$frame),]
Joint_dup <- Joint_frame %>% distinct(frame, .keep_all = TRUE)
Total <- Joint_dup %>% group_by(joint) %>% summarise(Count=n())
Result <- mutate(Total, Percent = Total$Count / sum(Total$Count)*100)
Result$joint <- as.character(Result$joint)
print(Result)
```

    ## # A tibble: 17 x 3
    ##    joint          Count Percent
    ##    <chr>          <int>   <dbl>
    ##  1 Left Ankle       171  18.0  
    ##  2 Left Ear          27   2.85 
    ##  3 Left Elbow        36   3.79 
    ##  4 Left Eye          36   3.79 
    ##  5 Left Hip           9   0.948
    ##  6 Left Knee         33   3.48 
    ##  7 Left Shoulder     45   4.74 
    ##  8 Left Wrist        19   2.00 
    ##  9 Neck              86   9.06 
    ## 10 Right Ankle      168  17.7  
    ## 11 Right Ear         41   4.32 
    ## 12 Right Elbow       65   6.85 
    ## 13 Right Eye         48   5.06 
    ## 14 Right Hip         38   4.00 
    ## 15 Right Knee        52   5.48 
    ## 16 Right Shoulder    43   4.53 
    ## 17 Right Wrist       32   3.37

## MarkDown Table

This code block allows you to obtain the “Result” dataframe inside a
table ready to be shared.

``` r
library(knitr)
```

    ## Warning: package 'knitr' was built under R version 4.0.5

``` r
Result_order <- Result[order(Result$Percent,decreasing=TRUE),]
kable(Result_order, caption = "Proportion of frames for each joint where it was the closest from gaze location")
```

| joint          | Count |    Percent |
| :------------- | ----: | ---------: |
| Left Ankle     |   171 | 18.0189673 |
| Right Ankle    |   168 | 17.7028451 |
| Neck           |    86 |  9.0621707 |
| Right Elbow    |    65 |  6.8493151 |
| Right Knee     |    52 |  5.4794521 |
| Right Eye      |    48 |  5.0579557 |
| Left Shoulder  |    45 |  4.7418335 |
| Right Shoulder |    43 |  4.5310854 |
| Right Ear      |    41 |  4.3203372 |
| Right Hip      |    38 |  4.0042150 |
| Left Elbow     |    36 |  3.7934668 |
| Left Eye       |    36 |  3.7934668 |
| Left Knee      |    33 |  3.4773446 |
| Right Wrist    |    32 |  3.3719705 |
| Left Ear       |    27 |  2.8451001 |
| Left Wrist     |    19 |  2.0021075 |
| Left Hip       |     9 |  0.9483667 |

Proportion of frames for each joint where it was the closest from gaze
location

## Pie Chart

This code block creates a pie chart representation of our “Result”
dataframe.

``` r
slices <- c(Result$Percent)
lbls <- c(Result$joint)
pct <- round(slices/sum(slices)*100)
lbls <- paste(pct,"%",sep="")

pie(slices, labels = lbls , main = "Closest joint from gaze",col = rainbow(length(lbls)),cex=0.72)
legend("topright", c(Result$joint), cex = 0.8, fill = rainbow(length(lbls)))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

## Bar plot with Percentages

This code block creates a bar plot representation of our “Result”
dataframe.

``` r
ggplot(Result, aes(x=reorder(joint,Percent), y=Percent, fill=joint)) + 
  geom_bar(stat = "identity")+coord_flip()+theme(legend.position="none")
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

## Distance from human shape

This code block creates firstly the representation of the distance
between gaze and humain shape in function of frames and then a pie chart
representing the proportion of frames where the gaze is located on or
outside human shape.

``` r
ggplot(data=Shape_frame, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="Evolution of the distance between the gaze and the human shape", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
Human_count <- Shape_frame %>% filter(distance == 0) %>% count(distance)
Shape_Percent <- Human_count$n / nrow(Shape_frame)

slices2 <- c(Shape_Percent, 1-Shape_Percent)
lbls2 <- c("On human shape","Outside")
pct2 <- round(slices2/sum(slices2)*100)
lbls2 <- paste(pct2,"%",sep="")

pie(slices2, labels = lbls2 , main = "Gaze compared to human shape",col = rainbow(length(lbls2)),cex=0.72)
legend("topright", c("On human shape","Outside"), cex = 0.8, fill = rainbow(length(lbls2)))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-5-2.png)<!-- -->

## Timeplot for joints

This code block creates the representation of the distance between gaze
and every joint one by one in function of frames. Exactly like “Result”,
this distance is measured irrelevantly of wether the gaze is located on
or outside of the human shape.

``` r
Joint_RKnee <- Joint_dup %>% filter(joint=="Right Knee")
  ggplot(data=Joint_RKnee, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="x", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
Joint_LKnee <- Joint_dup %>% filter(joint=="Left Knee")
ggplot(data=Joint_LKnee, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="Evolution of the distance between the gaze and the left knee", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-6-2.png)<!-- -->

``` r
Joint_LAnkle <- Joint_dup %>% filter(joint=="Left Ankle")
ggplot(data=Joint_LAnkle, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="Evolution of the distance between the gaze and the left ankle", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-6-3.png)<!-- -->

``` r
Joint_RAnkle <- Joint_dup %>% filter(joint=="Right Ankle")
ggplot(data=Joint_RAnkle, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="Evolution of the distance between the gaze and the right ankle", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-6-4.png)<!-- -->

``` r
Joint_RShoulder <- Joint_dup %>% filter(joint=="Right Shoulder")
ggplot(data=Joint_RShoulder, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="Evolution of the distance between the gaze and the right shoulder", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-6-5.png)<!-- -->

``` r
Joint_LShoulder <- Joint_dup %>% filter(joint=="Left Shoulder")
ggplot(data=Joint_LShoulder, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="Evolution of the distance between the gaze and the left shoulder", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-6-6.png)<!-- -->

``` r
Joint_REar <- Joint_dup %>% filter(joint=="Right Ear")
ggplot(data=Joint_REar, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="Evolution of the distance between the gaze and the right ear", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-6-7.png)<!-- -->

``` r
Joint_LEar <- Joint_dup %>% filter(joint=="Left Ear")
ggplot(data=Joint_LEar, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="Evolution of the distance between the gaze and the left ear", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-6-8.png)<!-- -->

``` r
Joint_RHip <- Joint_dup %>% filter(joint=="Right Hip")
ggplot(data=Joint_RHip, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="Evolution of the distance between the gaze and the right hip", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-6-9.png)<!-- -->

``` r
Joint_LHip <- Joint_dup %>% filter(joint=="Left Hip")
ggplot(data=Joint_LHip, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="Evolution of the distance between the gaze and the left hip", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-6-10.png)<!-- -->

``` r
Joint_RElbow <- Joint_dup %>% filter(joint=="Right Elbow")
ggplot(data=Joint_RElbow, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="Evolution of the distance between the gaze and the right elbow", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-6-11.png)<!-- -->

``` r
Joint_LElbow <- Joint_dup %>% filter(joint=="Left Elbow")
ggplot(data=Joint_LElbow, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="Evolution of the distance between the gaze and the left elbow", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-6-12.png)<!-- -->

``` r
Joint_REye <- Joint_dup %>% filter(joint=="Right Eye")
ggplot(data=Joint_REye, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="Evolution of the distance between the gaze and the right eye", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-6-13.png)<!-- -->

``` r
Joint_LEye <- Joint_dup %>% filter(joint=="Left Eye")
ggplot(data=Joint_LEye, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="Evolution of the distance between the gaze and the left eye", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-6-14.png)<!-- -->

``` r
Joint_Neck <- Joint_dup %>% filter(joint=="Neck")
ggplot(data=Joint_Neck, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="Evolution of the distance between the gaze and the neck", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-6-15.png)<!-- -->

``` r
Joint_RWrist <- Joint_dup %>% filter(joint=="Right Wrist")
ggplot(data=Joint_RWrist, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="Evolution of the distance between the gaze and the right wrist", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-6-16.png)<!-- -->

``` r
Joint_LWrist <- Joint_dup %>% filter(joint=="Left Wrist")
ggplot(data=Joint_LWrist, aes(x=frame,y=distance)) + geom_line(colour="royalblue3")+labs(title="Evolution of the distance between the gaze and the left wrist", x="Frame",y="Distance")+theme(plot.title = element_text(color="royalblue3", hjust = 0.5))
```

![](RGazeClassify_files/figure-gfm/unnamed-chunk-6-17.png)<!-- -->
