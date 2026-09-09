import cv2
import joblib

from skimage.feature import hog

from extract_digit_roi import extract_digit_roi



# =====================================================
# 数字模型
# 使用 digit_mask 模型
# =====================================================

MODEL_PATH="./models/digit_classifier_mask.pkl"


classifier=joblib.load(
    MODEL_PATH
)



# =====================================================
# 颜色 -> 数字
# =====================================================

COLOR_TO_DIGIT={

    "red":1,
    "orange":2,
    "yellow":3,
    "green":4,
    "blue":5,
    "purple":6,
    "pink":7

}




# =====================================================
# HOG
# 与训练保持一致
# =====================================================

def extract_hog(img):


    feature=hog(

        img,

        orientations=9,

        pixels_per_cell=(8,8),

        cells_per_block=(2,2),

        block_norm="L2-Hys"

    )


    return feature





# =====================================================
# digit_mask预测
# =====================================================

def predict_digit(digit_mask):


    img=cv2.resize(

        digit_mask,

        (64,64)

    )


    feature=extract_hog(
        img
    )


    feature=feature.reshape(
        1,-1
    )


    pred=classifier.predict(
        feature
    )[0]


    confidence=0.5


    if hasattr(
        classifier,
        "predict_proba"
    ):

        confidence=max(
            classifier.predict_proba(feature)[0]
        )


    return int(pred),float(confidence)






# =====================================================
# 主函数
# =====================================================

def classify_candidates(
        image,
        topk=5
):


    candidates=extract_digit_roi(
        image
    )


    if len(candidates)==0:

        return []



    results=[]



    for c in candidates[:topk]:


        color=c["color"]

        color_digit=c["color_id"]



        # =============================
        # digit_mask模型
        # =============================

        model_digit,model_conf=predict_digit(
            c["digit_mask"]
        )



        # =============================
        # hole可信度
        # =============================


        hole_count=c["hole_count"]

        hole_ratio=c["hole_ratio"]



        if 2 <= hole_count <= 10:

            hole_count_score=1.0

        elif hole_count<=20:

            hole_count_score=0.5

        else:

            hole_count_score=0



        if 0.05<hole_ratio<0.8:

            hole_ratio_score=1.0

        elif hole_ratio<1.2:

            hole_ratio_score=0.5

        else:

            hole_ratio_score=0



        hole_score=(

            0.6*hole_count_score

            +

            0.4*hole_ratio_score

        )




        # =============================
        # 最终数字判断
        # =============================

        digit_score={}


        # 颜色权重
        digit_score[color_digit]=(
            0.7
        )


        # 模型权重
        digit_score[model_digit]=(
            digit_score.get(model_digit,0)
            +
            0.2*model_conf
        )


        # hole只影响置信度
        final_digit=max(
            digit_score,
            key=digit_score.get
        )



        # 是否一致

        digit_match=(
            model_digit==color_digit
        )



        # =============================
        # candidate总评分
        # =============================


        final_score=(

            0.65*c["score"]

            +

            0.20*model_conf

            +

            0.15*hole_score

        )



        results.append({

            "box":c["box"],


            "roi":c["roi"],


            "digit_mask":c["digit_mask"],



            # extract

            "extract_score":
                c["score"],



            # color

            "color":
                color,


            "color_digit":
                color_digit,



            # model

            "model_digit":
                model_digit,


            "model_conf":
                model_conf,



            # hole

            "hole_ratio":
                hole_ratio,


            "hole_count":
                hole_count,


            "hole_score":
                hole_score,



            # final

            "final_digit":
                final_digit,


            "digit_match":
                digit_match,


            "final_score":
                final_score

        })



    results.sort(

        key=lambda x:x["final_score"],

        reverse=True

    )


    return results