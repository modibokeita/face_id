from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify, flash
import mysql.connector
import cv2
from PIL import Image
import numpy as np
import os
import time
import json
import zlib
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from datetime import date
import face_recognition as fr
from base64 import b64encode, b64decode
from flask_cors import CORS
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["TEMPLATES_AUTO_RELOAD"] = True
api_key = '411faabec5fcbcbb24b4105fa3fe3c1c'
api_url = f'http://api.ipstack.com/{user_ip}?access_key={api_key}'
# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
cnt = 0
pause_cnt = 0
justscanned = False
mycursor = None
mydb = None
try:
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="Modibokeita",  # Provide the password if required
        database="registration"
    )
    mycursor = mydb.cursor()
except mysql.connector.Error as err:
    print(f"Error: {err}")
if mycursor is not None:
    print("Database connection established.")
else:
    print("Database connection not established.")

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier(
        "C:/Users/HP/Desktop/FlaskApp/resources/haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5

        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    cap = cv2.VideoCapture(0)

    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]

    img_id = lastid
    max_imgid = img_id + 100
    count_img = 0

    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = "dataset/" + nbr + "." + str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            print("Inserting into the database...")
            mycursor.execute("""INSERT INTO `img_dataset` (`img_id`, `img_student`) VALUES
                                ('{}', '{}')""".format(img_id, nbr))
            mydb.commit()

            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break
                cap.release()
                cv2.destroyAllWindows()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = "C:/Users/HP/Desktop/FlaskApp/dataset"

    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L');
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)

    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

    return redirect('/admin')


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():  # generate frame by frame from camera
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        global justscanned
        global pause_cnt

        pause_cnt += 1

        coords = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            if confidence > 70 and not justscanned:
                global cnt
                cnt += 1

                n = (100 / 30) * cnt
                # w_filled = (n / 100) * w
                w_filled = (cnt / 30) * w

                cv2.putText(img, str(int(n)) + ' %', (x + 20, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (153, 255, 255), 2, cv2.LINE_AA)

                cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color, 2)
                cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255), cv2.FILLED)

                mycursor.execute("select a.img_student, b.student_name, b.department"
                                 "  from img_dataset a "
                                 "  left join students b on a.img_student = b.student_id "
                                 " where img_id = " + str(id))
                row = mycursor.fetchone()
                snbr = row[0]
                sname = row[1]
                sdepartment = row[2]

                if int(cnt) == 30:
                    cnt = 0

                    mycursor.execute("insert into accs_hist (accs_date, accs_prsn) values('" + str(
                        date.today()) + "', '" + snbr + "')")
                    mydb.commit()

                    cv2.putText(img, sname + ' | ' + sdepartment, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (153, 255, 255), 2, cv2.LINE_AA)
                    time.sleep(1)

                    justscanned = True
                    pause_cnt = 0

            else:
                if not justscanned:
                    cv2.putText(img, 'UNKNOWN', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                if pause_cnt > 80:
                    justscanned = False

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0), "Face", clf)

        return img

    faceCascade = cv2.CascadeClassifier(
        "C:/Users/HP/Desktop/FlaskApp/resources/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    wCam, hCam = 400, 400

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break

@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    user_id = session.get('user_id')

    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Fetch the list of available courses for the user to enroll
    mycursor.execute("SELECT * FROM courses")
    courses = mycursor.fetchall()

    print("Form Data:", request.form)  # Debug statement

    if request.method == 'POST':
        for course in courses:
            course_id_key = 'course_id_{}'.format(course[0])
            enroll_button_key = 'enroll_button_{}'.format(course[0])

            if course_id_key in request.form and enroll_button_key in request.form:
                try:
                    # Extract course_id from the button name
                    course_id = int(request.form[course_id_key])
                except ValueError:
                    return "Invalid course_id. Please enter a valid integer."

                # Check if the user is already enrolled in the course
                mycursor.execute("SELECT * FROM enrollments WHERE user_id = %s AND course_id = %s",
                                (user_id, course_id))
                enrollment = mycursor.fetchone()

                if not enrollment:
                    # User is not enrolled in the course, so enroll them
                    mycursor.execute("INSERT INTO enrollments (user_id, course_id) VALUES (%s, %s)",
                                    (user_id, course_id))
                    mydb.commit()  # Commit the changes to the database
                    print("Enrollment successful!")  # Debug statement
                else:
                    # User is already enrolled in the course, handle accordingly (e.g., display a message)
                    return "You are already enrolled in this course."

    return render_template('enroll.html', courses=courses)

@app.route('/admin/view_enrollments')
def view_enrollments():
    # if 'user_id' not in session:
    #    return redirect(url_for('login'))

    # Check if the logged-in user is an administrator (you may have a role column in the users table)
    # admin_id = True  # Change this based on your actual logic for checking administrator status

    # if not admin_id:
    #    return redirect(url_for('login'))

    # Fetch all enrolled students for each course
    mycursor.execute("SELECT courses.course_id, enrollments.user_id, users.full_name "
                     "FROM courses "
                     "JOIN enrollments ON courses.id = enrollments.course_id "
                     "JOIN users ON enrollments.user_id = users.id")
    enrollments = mycursor.fetchall()
    print("Enrollments:", enrollments)
    return render_template('view_enrollments.html', enrollments=enrollments)
@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/mycourses')
def my_courses():
    # Assuming you have the user's ID stored in the session
    user_id = session.get('user_id')

    # Fetch the courses for the logged-in user (adjust the SQL query as needed)
    mycursor.execute("SELECT courses.course_id, courses.professor FROM courses "
                     "JOIN enrollments ON courses.id = enrollments.course_id "
                     "WHERE enrollments.user_id = %s", (user_id,))
    user_courses = mycursor.fetchall()
    print("User Courses:", user_courses)
    return render_template('my_courses.html', user_courses=user_courses)

@app.route('/register_user', methods=["GET", "POST"])
def register_user():
    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Assign inputs to variables
        username = request.form.get('username', '')
        full_name = request.form.get('name', '')
        # Ensure username was submitted
        if not username:
            return render_template("register.html", messager=1)

        # Query database for username
        mycursor.execute("SELECT * FROM `users` WHERE `username` = %s", (username, ))
        user_data = mycursor.fetchone()

        # Ensure username is not already taken
        if user_data is not None:
            return render_template("register.html", messager=5)

        # Query database to insert new user
        else:
            mycursor.execute("INSERT INTO users (username, full_name) VALUES (%s, %s)", (username, full_name))

            mydb.commit()
            new_user_id = mycursor.lastrowid
            if new_user_id:
                # Keep newly registered user logged in
                session["user_id"] = new_user_id

            # Flash info for the user
            flash(f"Registered as {username}")

            # Redirect user to homepage
            return redirect("/facesetup")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("register.html")


@app.route("/facereg", methods=["GET", "POST"])
def facereg():
    session.clear()
    if request.method == "POST":

        encoded_image = (request.form.get("pic") + "==").encode('utf-8')
        username = request.form.get("name")

        mycursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        username_data = mycursor.fetchone()

        if username_data is not None:
            user_id = username_data[0]  # Assuming 'id' is the first (and only) element in the tuple
            compressed_data = zlib.compress(encoded_image, 9)

            uncompressed_data = zlib.decompress(compressed_data)

            decoded_data = b64decode(uncompressed_data)

            new_image_handle = open('static/face/unknown/' + str(user_id) + '.jpg', 'wb')
            new_image_handle.write(decoded_data)
            new_image_handle.close()

            try:
                image_of_bill = fr.load_image_file('static/face/' + str(user_id) + '.jpg')
                bill_face_encoding = fr.face_encodings(image_of_bill)[0]

                unknown_image = fr.load_image_file('static/face/unknown/' + str(user_id) + '.jpg')
                unknown_face_encoding = fr.face_encodings(unknown_image)[0]

                # Compare faces
                results = fr.compare_faces([bill_face_encoding], unknown_face_encoding)

                if results[0]:
                    mycursor.execute("SELECT id, username FROM users WHERE username = %s", (username,))
                    user_data = mycursor.fetchone()
                    if user_data:
                        user_id, username = user_data
                        session['user_id'] = user_id
                        session['username'] = username
                        flash(f"Welcome back, {username}!")
                    username = session.get('username')
                    if username:
                        return render_template('home.html', username=username)
                    flash("Please log in to access the home page.")
                    return redirect('/home')


            except Exception as e:
                print(e)
                return render_template("camera.html", message=5)
        else:
            return render_template("camera.html", message=4)
    return render_template("camera.html", message=3)

@app.route("/facesetup", methods=["GET", "POST"])
def facesetup():
    if request.method == "POST":

        encoded_image = (request.form.get("pic") + "==").encode('utf-8')
        mycursor.execute("SELECT id FROM `users` WHERE `id` = %s", (session["user_id"],))

        result = mycursor.fetchone()
        if result:
            id_ = result[0]
            compressed_data = zlib.compress(encoded_image, 9)
            uncompressed_data = zlib.decompress(compressed_data)
            decoded_data = b64decode(uncompressed_data)
            new_image_handle = open('static/face/' + str(id_) + '.jpg', 'wb')

            new_image_handle.write(decoded_data)
            new_image_handle.close()
            image_of_bill = fr.load_image_file('static/face/' + str(id_) + '.jpg')

            try:
                bill_face_encoding = fr.face_encodings(image_of_bill)[0]
            except IndexError:
                return render_template("face.html", message=1)
            return redirect("/login")
        else:
            # Handle the case where no user with the given id is found
            return render_template("face.html", message=2)

    else:
        return render_template("face.html")

@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/')
def home_page():
    return render_template('home2.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route('/admin')
def admin():
    if mycursor is not None:
        mycursor.execute("select student_id, student_name, studentID, department, email, student_added from students")
        data = mycursor.fetchall()
        
        mycursor.execute("SELECT id, course_name, course_id, professor FROM courses")
        course_data = mycursor.fetchall()   
        return render_template('index.html', data=data, course_data=course_data)
    else:
        return "Database connection not established."
   

@app.route('/addprsn')
def addprsn():
    mycursor.execute("select ifnull(max(student_id) + 1, 101) from students")
    row = mycursor.fetchone()
    nbr = row[0]
    # print(int(nbr))

    return render_template('addprsn.html', newnbr=int(nbr))


@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    snbr = request.form.get('txtnbr')
    sname = request.form.get('txtname')
    sdepartment = request.form.get('txtdep')
    semail = request.form.get('txtemail')
    suid = request.form.get('txtnumber')
    
    mycursor.execute("""INSERT INTO `students` (`student_id`, `student_name`, `studentID`, `department`, `email`) VALUES
                    ('{}', '{}', '{}','{}', '{}')""".format(snbr, sname, suid, sdepartment, semail))
    mydb.commit()

    return redirect(url_for('vfdataset_page', prs=snbr))

@app.route('/addcourses')
def addcourses():
    mycursor.execute("SELECT MAX(id) + 1 FROM courses")
    row = mycursor.fetchone()
    nbid = row[0] if row[0] is not None else 1  # Handle the case when the table is empty

    return render_template('add_courses.html', newcoure=int(nbid))


@app.route('/addcourse_submit', methods=['POST'])
def addcourse_submit():
    cnbr = request.form.get('txtcourseid')
    cid = request.form.get('txtid')
    cname = request.form.get('txtcname')
    prof = request.form.get('txtprof')

    
    mycursor.execute("""INSERT INTO `courses` (`id`, `course_name`, `course_id`, `professor`) VALUES
                    ('{}', '{}', '{}','{}')""".format(cnbr, cname, cid, prof))
    mydb.commit()

    return redirect(url_for('admin'))

@app.route('/delete_prsn')
def delete_prsn():
    snbr = request.form.get('txtnbr')

    mycursor.execute("DELETE FROM `students` WHERE `student_id` = '{}'".format(snbr))
    mydb.commit()

    return render_template('delete_prsn.html')

@app.route('/delete_prsn_submit', methods=['POST'])
def delete_prsn_submit():
    snbr = request.form.get('txtnbr')
    delete_prsn()
    return redirect(url_for('admin'))


@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('gendataset.html', prs=prs)


@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fr_page')
def fr_page():
    """Video streaming home page."""
    mycursor.execute("select a.accs_id, a.accs_prsn, b.student_name, b.department, a.accs_added "
                     "  from accs_hist a "
                     "  left join students b on a.accs_prsn = b.student_id "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()

    return render_template('fr_page.html', data=data)


@app.route('/countTodayScan')
def countTodayScan():
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="Modibokeita",  # Provide the password if required
            database="registration"
        )
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    else:
        mycursor = mydb.cursor()

    mycursor.execute("select count(*) "
                     "  from accs_hist "
                     " where accs_date = curdate() ")
    row = mycursor.fetchone()
    rowcount = row[0]

    return jsonify({'rowcount': rowcount})


@app.route('/loadData', methods=['GET', 'POST'])
def loadData():
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="Modibokeita",  # Provide the password if required
            database="registration"
        )
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    else:
        mycursor = mydb.cursor()

    mycursor.execute("select a.accs_id, a.accs_prsn, b.student_name, b.department, date_format(a.accs_added, '%H:%i:%s') "
                     "  from accs_hist a "
                     "  left join students b on a.accs_prsn = b.student_id "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()

    return jsonify(response=data)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
