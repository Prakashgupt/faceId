
from flask import Flask, render_template, request, jsonify, flash

from werkzeug.utils import secure_filename
from modelproceesing import *

from  frameExtracter import *
from firebase_config import *
UPLOAD_FOLDER = './existing_user'
UPLOAD_FOLDER1 = './Train'
UPLOAD_FOLDER2='./TempImg'
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER1'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = "312d9ab81ec3eeabb1fd435a5bf38700"









@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route('/checkFace', methods=['POST'])
def success():

    if request.method == 'POST':
        # print(request.files)
        # print(request.form)
        if 'image' not in request.files:
            flash('No file part')
            return
        file = request.files['image']
        print(file)
        if file.filename == '':
            flash('No selected file')
            return
        # path = os.path.join(parent_dir, directory)
        title = request.form.get("title")

        # im_path = "Image/" + str(title) + "/" + str(title)+ ".jpg"
        # storage1.child(im_path).put(file)


        path=os.path.join(UPLOAD_FOLDER2,title)
        os.mkdir(path);
        filename = secure_filename(file.filename)
        image_path=os.path.join(path,filename)
        file.save(image_path)
        print(image_path)
        # # taking photo in data base
        # os.chdir(path)
        # dst_dir = "existing_user"
        # for f in os.listdir():
        #     shutil.copy(f, dst_dir)


        # path1 = os.path.join(UPLOAD_FOLDER, "Photo")
        # if not os.path.exists(path1):
        #     os.mkdir(path1)
        # filename1 = secure_filename(file.filename)
        # image_path1 = os.path.join(path1, filename1)
        # file.save(image_path1)
        # print(image_path)










        eye_img=chek1(image_path,str(title))
        if  eye_img is None:
            shutil.rmtree('./TempImg')
            os.mkdir('./TempImg')
            return jsonify({'output': 'unsuccess',
                            # 'file name' : str(file.filename),
                            #                     # 'image_url':storage1.child("Image/"+title+"/"+title+".jpg").get_url(None),
                            'eye_url': "https://firebasestorage.googleapis.com/v0/b/facedectectionst.appspot.com/o/face-not-found.jpeg?alt=media&token=e1495d6e-a736-43bf-9b10-757db01f190f"
                                    })


            # img_path1 = os.path.join(folder_path, img_name)
            # print(img_path1)

        # storage1.child(img_path1).put(imag)
        # url = storage1.child(img_path1).get_url(None)
        # print(url)


        # if  eye_img is None :
        #     return jsonify({'output': 'unsuccess',
        #                     # 'file name' : str(file.filename),
        #                     # 'image_url':storage1.child("Image/"+title+"/"+title+".jpg").get_url(None),
        #                     'eye_url': "face Not found"
        #                     })
        img=sol(ch=None)
        if img is None :
            shutil.rmtree('./TempImg')
            os.mkdir('./TempImg')
            shutil.rmtree('./Checkeye')
            os.mkdir('./Checkeye')
            return jsonify({'output': 'unsuccess',
                            # 'file name' : str(file.filename),
                            #                     # 'image_url':storage1.child("Image/"+title+"/"+title+".jpg").get_url(None),
                           'eye_url': "https://firebasestorage.googleapis.com/v0/b/facedectectionst.appspot.com/o/error.png.jpeg?alt=media&token=84dc24e6-63df-49d2-9d16-4e0d881fe141"
                            })

        # eye_url=upload_blob("Train/" +str(img) +"/extracted_eye.jpg")
        image=upload_blob(img)
        shutil.rmtree('./Checkeye')
        os.mkdir('./Checkeye')
        shutil.rmtree('./TempImg')
        os.mkdir('./TempImg')
        print(image)
        print("success")
        return  jsonify({'output':'success',
                         # 'file name' : str(file.filename),
                         # 'image_url':storage1.child("Image/"+title+"/"+title+".jpg").get_url(None),
                         'eye_url' : image
                                  })


@app.route('/addFace', methods=['POST'])
def add_face():
    if request.method == 'POST':
        title = request.form.get("title")
        for i in range(4):
            file1 = request.files['file'+str(i)]
            print(file1)
            im_path1 = "Image/" + title + "/" + str(i)+".jpg"
            # storage1.child(im_path1).put(file1)
            path = os.path.join(UPLOAD_FOLDER1, title)
            if not os.path.exists(path):
                os.mkdir(path)

            filename = secure_filename(file1.filename)
            image_path = os.path.join(path, filename)
            file1.save(image_path)

            # url1 = storage1.child(im_path1).get_url(None)
            eye=chek(image_path, title,str(i)+'.jpg')
            # resizePhoto(image_path)

        folder_created = os.path.join(UPLOAD_FOLDER1, title)
        if len(os.listdir(folder_created)) == 0:
            shutil.rmtree(folder_created)
            print('folder destroyed!')
            return jsonify({'result':'unsuccess , please sign up again'})


            # eye_img= upload_blob("Train/" + title + "/"+str(i)+".jpg")


    return jsonify({'result':'success'})












if __name__ == '__main__':
    app.run(debug=True,port=5005)
