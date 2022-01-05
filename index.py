from flask import Flask,render_template,request
from keras.preprocessing.image import load_img,img_to_array
import keras 
import numpy as np
import os
# the model
model= keras.models.load_model("version_1")
app=Flask(__name__)
@app.route("/",methods=["GET"])
async def home_page():
   return  render_template('home.html')

@app.route("/",methods=["POST"])
async def predict():
    image = request.files['imagefile']
    global path
    path= "./static/images/"+image.filename
    image.save(path)
    img=load_img(path,target_size=(256,256))
    img=img_to_array(img).reshape(1,256,256,3)
    classes=["Early Blight","Late Blight","Healthy"]
    prediction= model.predict(img)
    acc=str(round(np.max(prediction)*100,2))+" %"
    predicted_class= classes[np.argmax(prediction)]
    return render_template('home.html',image_path=path,classname=predicted_class,accuracy=acc)
    
    
if __name__=="__main__":
    app.run(debug=True,port=3000)