!pip install gradio lime

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from lime import lime_image
from keras.applications import inception_v3 as inc_net
from keras.applications.imagenet_utils import decode_predictions

inet_model = inc_net.InceptionV3()
explainer = lime_image.LimeImageExplainer()

def predict_and_explain(input_image):
    # Gradio gives us a numpy array directly — resize + preprocess like transform_img_fn did
    img = np.array(input_image.resize((299, 299)))
    x = np.expand_dims(img, axis=0)
    x = inc_net.preprocess_input(x.astype('float'))

    # Prediction
    preds = inet_model.predict(x)
    top5 = decode_predictions(preds)[0]
    pred_text = "\n".join([f"{label}: {round(float(prob)*100, 2)}%" for (_, label, prob) in top5])

    # LIME explanation
    explanation = explainer.explain_instance(
        x[0].astype('double'), inet_model.predict,
        top_labels=5, hide_color=0, num_samples=500, batch_size=50
    )

    def get_vis(**kwargs):
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], **kwargs)
        return mark_boundaries(temp / 2 + 0.5, mask)

    vis1 = get_vis(positive_only=True, num_features=5, hide_rest=True)
    vis2 = get_vis(positive_only=True, num_features=8, hide_rest=False)
    vis3 = get_vis(positive_only=False, num_features=10, hide_rest=False)
    vis4 = get_vis(positive_only=False, num_features=1000, hide_rest=False, min_weight=0.035)

    return pred_text, vis1, vis2, vis3, vis4

demo = gr.Interface(
    fn=predict_and_explain,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Textbox(label="Top 5 Predictions"),
        gr.Image(label="Top feature only (hidden background)"),
        gr.Image(label="Top features (with background)"),
        gr.Image(label="Positive + negative regions"),
        gr.Image(label="Weighted threshold view (min_weight=0.035)")
    ],
    title="InceptionV3 + LIME: Why did the model predict this?",
    description="Upload an image and see which regions InceptionV3 used to make its top prediction, explained via LIME."
)

demo.launch()