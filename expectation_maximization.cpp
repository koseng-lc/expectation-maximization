#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>

#define TRAINING 0
#define BUAT_TABEL 0
#define PATH_DATA "/mnt/E/alfarobi/color_distribution.xml"
#if TRAINING == 1
    #define MODE FileStorage::WRITE
#elif BUAT_TABEL == 1
    #define MODE FileStorage::READ
#else
    #define MODE FileStorage::READ
    #undef PATH_DATA
    #define PATH_DATA "/mnt/E/alfarobi/tabel_warna.xml"
#endif

using namespace cv;
using namespace std;

//-- RGB
int dimensi = 3;

//-- jumlah data training
int jml_data;

//-- jumlah komponen / kluster
int jml_komponen = 3;

//-- konstanta
double fraksi;

//-- membership weight
std::vector<double> memw;

//-- mixture weight
std::vector<double> mixw(jml_komponen);

//std::vector<Mat> meank(jml_komponen, Mat(dimensi, 1, CV_64FC1));
std::vector<Mat> meank(jml_komponen);

std::vector<Mat> covk(jml_komponen);

int uk_cuplikan = 150;

#if TRAINING == 1 || BUAT_TABEL == 1

void initGMM(){

    memw.resize(jml_data * jml_komponen);

    //Hijau - Lapang
//    meank[0].at<double>(0) = 90>>2;
//    meank[0].at<double>(1) = 196>>2;
//    meank[0].at<double>(2) = 158>>2;

//    //Putih - Bola
//    meank[1].at<double>(0) = 194>>2;
//    meank[1].at<double>(1) = 197>>2;
//    meank[1].at<double>(2) = 197>>2;

//    //Sisa
//    meank[2].at<double>(0) = 10>>2;
//    meank[2].at<double>(1) = 3>>2;
//    meank[2].at<double>(2) = 9>>2;

    covk[0] = Mat::eye(dimensi, dimensi, CV_64FC1);
    covk[0] *= 16.0;

    covk[1] = Mat::eye(dimensi, dimensi, CV_64FC1);
    covk[1] *= 16.0;

    covk[2] = Mat::eye(dimensi, dimensi, CV_64FC1);
    covk[2] *= 16.0;

    //for(int i=0;i<jml_komponen;i++){
        //cout<<"Mean Awal -"<<i+1<<meank[i]<<endl;
//        covk[i].at<double>(0) = 1000;
//        covk[i].at<double>(4) = covk[i].at<double>(0);
//        covk[i].at<double>(8) = covk[i].at<double>(0);
    //}

    fraksi = 1.0 / pow((double)M_PI * 2.0, (double)dimensi / 2.0);
}

inline double probDF(const Mat& _x, const Mat& _mean, const Mat& _cov){

    double det = determinant(_cov);

    if(det <= .0){
        throw "Over-fitting!!!";
    }

    det = 1.0 / sqrt(det);

    Mat diff = _x - _mean;

    Mat diff_t = diff.t();

    Mat cov_inv = _cov.inv();

    Mat temp = diff_t * cov_inv;

    temp *= diff;

    double eksponen = exp(-.5 * temp.at<double >(0));

    return fraksi * det * eksponen;
}

//double prediksiProb(Mat &input){
//    double prob=0;
//    for(int i=0;i<jml_komponen;i++){
//        prob+=mixw[i]*probDF(input,meank[i],covk[i]);
//    }
//    return prob;
//}
#else
void cropLuar(Mat &mat_thresh,Mat &out){

    Mat hvs = Mat::zeros(mat_thresh.size(),CV_8UC1);

    vector<vector<Point > > contours;

    vector<Point > titik_kontur;

    //vector<int > batas_lapangan(input.cols);

    findContours(mat_thresh,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);

    for(size_t i=0;i<contours.size();i++){
        if((contourArea(contours[i])>500)){
            titik_kontur.insert(titik_kontur.end(),contours[i].begin(),contours[i].end());
        }
    }

    if(titik_kontur.size()){
        vector<Point > titik_hull;
        vector<vector<Point > > contours2;
        convexHull(titik_kontur,titik_hull);
        contours2.push_back(titik_hull);
        //Kontur penuh
        drawContours(hvs,contours2,0,Scalar::all(255),CV_FILLED);
    }
    hvs.copyTo(out);
}
#endif

#if TRAINING == 1

inline Mat pikselKeMatriks(const Vec3b& v){

    Mat d(jml_komponen, 1, CV_64FC1);
    d.at<double >(0) = v[0];
    d.at<double >(1) = v[1];
    d.at<double >(2) = v[2];

    return d;
}

//double cekKonvergen(Mat dataset){
//    double val=0;
//    int flag=0;
//    for(int i=0;i<jml_data;i++){
//        Mat d = pikselKeMatriks(dataset.at<Vec3b >(i));
//        double logaritma=0;
//        for(int j=0;j<jml_komponen;j++){
//            double prob=probDF(d,meank[j],covk[j]);
//            if(prob<0.0){
//                flag=1;
//                break;
//            }
//            logaritma += mixw[j]*prob;
//        }
//        if(flag==1)break;
//        logaritma = log10(logaritma);
//        val+=logaritma;
//    }
//    if(flag==1)return 0.0;
//    return val;
//}


int EM(Mat& dataset, int iterasi){
    // 1 - Berhasil
    //-1 - Gagal
    Mat blank_mean = Mat::zeros(dimensi, 1, CV_64FC1);
    Mat blank_cov = Mat::zeros(dimensi, dimensi, CV_64FC1);

    double log_like = .0;
    double prev_log_like = .0;

    for(int iter = 0; iter < iterasi; iter++){
        //Cek Konvergen
        //log_like = cekKonvergen(dataset);

        log_like = .0;

        for(int i = 0; i < jml_data; i++){            

            Mat d = pikselKeMatriks(dataset.at<Vec3b >(i));

            double logaritma = .0;

            for(int j = 0; j < jml_komponen; j++){

                double prob;

                try{

                    prob = probDF(d, meank[j], covk[j]);

                }catch(const char* e){

                    cerr << e << endl;

                    return -1;
                }

                logaritma += mixw[j] * prob;
            }

            logaritma = log10(logaritma);

            log_like += logaritma;
        }

        cout << iter + 1 << ". LOG:" << log_like << endl;

        if(log_like != log_like || log_like == .0) return -1;

        if(iter != 0 && fabs(log_like - prev_log_like) <= DBL_EPSILON) break;

        prev_log_like = log_like;

        // E-step

        for(int i = 0; i < jml_data; i++){

            Mat d = pikselKeMatriks(dataset.at<Vec3b >(i));

            double total_memw = .0;
            double temp_memw = .0;

            for(int j = 0; j < jml_komponen; j++){

                double prob = .0;

                try{

                    prob = probDF(d, meank[j], covk[j]);

                }catch(const char* e){

                    cout << e << endl;

                    return -1;
                }

                temp_memw = mixw[j] * prob;

                memw[(i * jml_komponen) + j] = temp_memw;

                total_memw += temp_memw;
            }


            if(total_memw > .0){

                for(int j = 0 ; j < jml_komponen; j++)
                    memw[(i * jml_komponen) + j] /= total_memw;
            }
        }        

        // M-step

        for(int i = 0; i < jml_komponen; i++){

            double nk = .0;

            for(int j = 0; j < jml_data; j++){

                nk += memw[j * jml_komponen + i];
            }

            mixw[i] = nk / jml_data;

            blank_mean.copyTo(meank[i]);

            for(int j = 0; j < jml_data; j++){

                Mat d = pikselKeMatriks(dataset.at<Vec3b >(j));

                meank[i] = meank[i] + memw[j * jml_komponen + i] * d;
            }

            if(nk > .0) meank[i] = (1.0 / nk) * meank[i];

            blank_cov.copyTo(covk[i]);

            Mat diff;
            Mat diff_t;
            Mat temp;

            for(int j = 0; j < jml_data; j++){

                Mat d = pikselKeMatriks(dataset.at<Vec3b >(j));

                diff = d - meank[i];
                diff_t = diff.t();
                temp = diff * diff_t;
                temp = temp * memw[j * jml_komponen + i];
                covk[i] = covk[i] + temp;
            }

            if(nk > .0) covk[i] = (1.0 / nk) * covk[i];
        }
    }

    return 1;
}

//======================== Inteface ======================

bool klik_mouse = false;
bool imshow_flag = false;
bool klik_kanan = false;
int tlx = 0, tly = 0;
int brx = 0, bry = 0;
int mouse_x = 0, mouse_y = 0;

void callBack(int event, int x, int y, int flags, void* userdata){
    if(event == EVENT_LBUTTONDOWN){
        if(!klik_mouse){
            tlx = x;
            tly = y;
        }else{
            brx = x;
            bry = y;
            if(brx - tlx > uk_cuplikan) brx = tlx + uk_cuplikan;
            else if(brx - tlx < -uk_cuplikan) brx= tlx - uk_cuplikan;
            if(bry - tly > uk_cuplikan) bry = tly + uk_cuplikan;
            else if(bry - tly < -uk_cuplikan) bry = tly - uk_cuplikan;
            imshow_flag = true;
        }
        klik_mouse = !klik_mouse;
    }else if(event == EVENT_MOUSEMOVE){
        mouse_x = x;
        mouse_y = y;
    }else if(event == EVENT_RBUTTONDOWN){
        klik_kanan = true;
        tlx = x;
        tly = y;
    }
}

#endif

#if BUAT_TABEL == 1
inline int prediksiLabel(const Mat& input){
    int label = jml_komponen;
    double maks = 0;
    for(int i = 0; i < jml_komponen; i++){
        double prob = mixw[i] * probDF(input, meank[i], covk[i]);
        if(prob > maks && prob > DBL_EPSILON){
            label = i;
            maks = prob;
        }
    }
    return label;
}
#endif

int main(){

    VideoCapture vc("/mnt/E/alfarobi/recorded/video3.avi");

    if(!vc.isOpened()){ cerr << "Error coy!!" << endl; return -1; }

    FileStorage fs = FileStorage(PATH_DATA, MODE);

    //vc.set(CV_CAP_PROP_POS_MSEC,4000);
    
#if TRAINING == 1

    vector<Mat > data_tr;
    
    int sz = 0;
    
    for(int i = 0; i < jml_komponen; i++){

        meank[i] = Mat::zeros(dimensi, 1, CV_64FC1);
        mixw[i] = 1.0 / (double)jml_komponen;
    }

    while(1){
        Mat frame;
        vc >> frame;
        resize(frame, frame, Size(320,240));
        //flip(frame,frame,-1);
        
        imshow("FRAME",frame);
        int c = waitKey(33);
        
        if(c == 32){

            namedWindow("Captured", CV_WINDOW_NORMAL);

            setMouseCallback("Captured", callBack, NULL);

            while(1){

                Mat copy;                
                frame.copyTo(copy);

                if(klik_mouse){

                    if(mouse_y - tly > uk_cuplikan) mouse_y = tly + uk_cuplikan;
                    else if(mouse_y - tly < -uk_cuplikan) mouse_y = tly - uk_cuplikan;
                    if(mouse_x - tlx > uk_cuplikan) mouse_x = tlx + uk_cuplikan;
                    else if(mouse_x - tlx < -uk_cuplikan) mouse_x = tlx - uk_cuplikan;

                    line(copy,Point(tlx, tly), Point(mouse_x, tly), Scalar(0, 0, 255), 1);
                    line(copy,Point(tlx, tly), Point(tlx, mouse_y), Scalar(0, 0, 255), 1);
                    line(copy,Point(mouse_x, tly), Point(mouse_x, mouse_y), Scalar(0, 0, 255), 1);
                    line(copy,Point(tlx, mouse_y), Point(mouse_x, mouse_y), Scalar(0, 0, 255), 1);
                }

                imshow("Captured", copy);

                int ch = waitKey(20);

                if(imshow_flag){

                    imshow_flag = false;

                    int temp_tlx = tlx;
                    int temp_tly = tly;

                    if(tlx>brx)temp_tlx = brx;
                    if(tly>bry)temp_tly = bry;

                    Rect r(temp_tlx, temp_tly, abs(tlx-brx), abs(tly-bry));
                    Mat roi(frame, r);

                    imshow("ROI", roi);

                    while(1){

                        int ch2 = waitKey(20);

                        if(ch2 == 27){ cout << "Gambar tidak disimpan" << endl; break;}
                        else if(ch2 == 32){
                            sz += roi.cols * roi.rows;

                            data_tr.push_back(roi);

                            cout << "Gambar disimpan !!" << endl;
                            break;
                        }
                    }
                }else if(ch == 27){

                    if(klik_mouse){

                        klik_mouse=false;
                    }else{

                        break;
                    }

                }else {

                    for(int i = 0; i < jml_komponen; i++){

                        if(ch == (49 + i)){

                            meank[i].at<double >(0) = frame.at<Vec3b >(mouse_y, mouse_x)[0] >> 2;
                            meank[i].at<double >(1) = frame.at<Vec3b >(mouse_y, mouse_x)[1] >> 2;
                            meank[i].at<double >(2) = frame.at<Vec3b >(mouse_y, mouse_x)[2] >> 2;

                            cout<< "Inisialisasi Mean-" << i+1 << " Disimpan !!" <<endl;

                            cout<< meank[i] << endl;

                            break;
                        }
                    }
                }
            }
        }else if(c == 27){

            break;
        }
    }

    vc.release();

    destroyAllWindows();

    if(!data_tr.size()){

        cerr << "Data Training tidak bisa kosong !!!" << endl;

        return -1;
    }

    Mat dataset = Mat(sz, 1, CV_8UC3);

    //data_vector = new Mat[sz];
    //parameter histogram
//    int channels[3] = {0,1,2};
//    int hist_size[3] = {256,256,256};
//    float hranges[2] = {0,255.0};
//    const float* ranges[3] = {hranges,hranges,hranges};

//    int channels[1] = {0};
//    int hist_size[1] = {256};
//    float hranges[2] = {0,255.0};
//    const float* ranges[1] = {hranges};
    //==================================================
    int h = 0;

    for(size_t x = 0; x < data_tr.size(); x++){

        /*Mat hist[2];//(3,hist_size,CV_64F);
        Mat hsv;
        Mat thresh;
        vector<Mat > data_split;

        //Nyari warna dominan

        cvtColor(data_tr[x],hsv,CV_BGR2HSV);
        split(hsv,data_split);
        cout<<"Baris:"<<data_tr[x].rows<<";Kolom:"<<data_tr[x].cols<<endl;

        calcHist(&data_split[0],1,channels,Mat(),hist[0],1,hist_size,ranges);
        calcHist(&data_split[1],1,channels,Mat(),hist[1],1,hist_size,ranges);

        Point min_idx,maks_idx[2];
        double minv,maksv[2];

        minMaxLoc(hist[0],&minv,&maksv[0],&min_idx,&maks_idx[0]);
        minMaxLoc(hist[1],&minv,&maksv[1],&min_idx,&maks_idx[1]);

        cout<<maks_idx[0]<<endl;
        cout<<maksv[0]<<endl;

        cout<<maks_idx[1]<<endl;
        cout<<maksv[1]<<endl;

        inRange(hsv,Scalar(maks_idx[0].y,(maks_idx[1].y>10)?maks_idx[1].y-10:maks_idx[1].y,0),Scalar(maks_idx[0].y,(maks_idx[1].y<245)?maks_idx[1].y+10:maks_idx[1].y,255),thresh);
        inRange(hsv,Scalar(maks_idx[0].y,0,0),Scalar(maks_idx[0].y,255,255),thresh);

        imshow("THRESH",thresh);
        imshow("HSV",hsv);

        uchar* thr_data = thresh.data;*/

        uchar* smpl_data = data_tr[x].data;

        int baris = data_tr[x].rows;
        int kolom = data_tr[x].cols;
        int kanal = data_tr[x].channels();

        for(int i=0;i<baris;i++){

            for(int j=0;j<kolom;j++){

                //if(thr_data[i*kolom+j]){
                    int B = smpl_data[(i*kolom+j)*kanal+0];
                    int G = smpl_data[(i*kolom+j)*kanal+1];
                    int R = smpl_data[(i*kolom+j)*kanal+2];

                    B = B >> 2;
                    G = G >> 2;
                    R = R >> 2;
//                    int push=1;
//                    for(size_t k=0;k<data_vector.size();k++){
//                        if(!equal(data_vector[k].begin<uchar>(),data_vector[k].end<uchar>(),temp.begin<uchar>())){push=0;break;}
//                    }
                    //data_vector.push_back((Mat_<double>(dimensi,1) << B, G, R));
                    //data_vector[h] =  (Mat_<double>(dimensi,1) << B, G, R);
                    dataset.at<Vec3b > (h)[0] = B;
                    dataset.at<Vec3b > (h)[1] = G;
                    dataset.at<Vec3b > (h)[2] = R;
                    h++;
                //}
            }
        }
        //waitKey(0);
    }

    jml_data = sz;

    initGMM();

    cout << "Jumlah Data:" << jml_data << endl;

    cout << "Training....." << endl;

    int flag = EM(dataset, 100);
    
    if(flag < 0.0){

        cout << "Training gagal" << endl;

    }else{

        cout << "Training berhasil" << endl;
        cout << "Simpan..." << endl;

        for(int i = 0; i < jml_komponen; i++){

            cout << "Mean-"<< i + 1 << " : " << meank[i] <<endl;
            cout << "Cov-" << i+1 << " : " << covk[i] << endl;
            cout << "Bobot-"<< i + 1<< " : " << mixw[i] <<endl;

            stringstream ss;
            ss << i+1;
            fs << "Mean-" + ss.str() << meank[i];
            fs << "Cov-" + ss.str() << covk[i];
            fs << "Bobot-" + ss.str() << mixw[i];

        }
    }

    fs.release();

#else
    //EKSEKUSI !!!!
    Mat lut_gmm;
#if BUAT_TABEL == 1

    initGMM();

    for(int i = 0; i < jml_komponen; i++){
        stringstream ss;
        ss << i+1;
        fs["Mean-"+ss.str()] >> meank[i];
        fs["Cov-"+ss.str()] >> covk[i];
        fs["Bobot-"+ss.str()] >> mixw[i];
    }

    fs.release();

    FileStorage fs2 = FileStorage("/mnt/E/alfarobi/tabel_warna.xml", FileStorage::WRITE);

    lut_gmm = Mat::zeros(pow(2,18), 1, CV_8UC1);

    uchar* lut_gmm_data = lut_gmm.data;

    cout << "Buat Tabel Warna...." << endl;

    for(int i = 0; i < lut_gmm.rows; i++){

        int R = (0b000000000000111111 & i);
        int G = (0b000000111111000000 & i) >> 6;
        int B = (0b111111000000000000 & i) >> 12;

        lut_gmm_data[i] = prediksiLabel((Mat_<double>(dimensi, 1) << B,G,R));
    }

    cout << "Beres!!" << endl;
    write(fs2, "Tabel_Warna", lut_gmm);
    fs2.release();

#else

    fs["Tabel_Warna"] >> lut_gmm;
    fs.release();

    while(1){

        Mat frame;
        Mat cpy;

        vc >> frame;
        Mat lapang;

        //flip(frame,frame,-1);

        Mat kont_hijau = Mat::zeros(frame.size(), CV_8UC1);
        Mat kont_putih = Mat::zeros(frame.size(), CV_8UC1);

        frame.copyTo(cpy);

        double dT = getTickCount();

        int kolom = frame.cols;
        int baris = frame.rows;
        int knl = frame.channels();

        uchar* frame_data = frame.data;

        for(int i=0;i<baris;i++){
            for(int j=0;j<kolom;j++){
                int idx = (i*kolom+j)*knl;
                int B = frame_data[idx + 0] >> 2;
                int G = frame_data[idx + 1] >> 2;
                int R = frame_data[idx + 2] >> 2;
                int idx_tabel = B << 12 | G << 6 | R;
                switch(lut_gmm.at<uchar > (idx_tabel)){
                    case 0:frame_data[idx] = 0; frame_data[idx+1] = 255; frame_data[idx + 2] = 0; kont_hijau.at<uchar > (idx / knl) = 255; break;
                    case 1:frame_data[idx] = 0; frame_data[idx+1] = 0; frame_data[idx + 2] = 255; kont_putih.at<uchar > (idx / knl) = 255; break;
                    case 2:frame_data[idx] = 255; frame_data[idx+1] = 255; frame_data[idx + 2] = 255;break;
                    default:break;
                }
            }
        }
        cropLuar(kont_hijau, lapang);

        dT = (getTickCount() - dT) / getTickFrequency();
        cout << "dT : " << dT << endl;

        cvtColor(lapang, lapang, CV_GRAY2BGR);
        bitwise_and(lapang, frame, frame);
        imshow("CPY", cpy);
        imshow("FRAME", frame);
        if(waitKey(33) == 27) break;
    }

#endif

#endif

    return 0;

}

/*
PROGRAM TES AJAHHH !!!!
Mat img=imread("/media/lintang/563C3F913C3F6ADF/Media/01.jpg");
resize(img,img,Size(img.rows/2,img.cols/2));
vector<Mat > data_flat;
uchar* img_data = img.data;
int kol = img.cols;
int bar = img.rows;
int knl = img.channels();
for(int i=0;i<bar;i++){
    for(int j=0;j<kol;j++){
        Mat tmp(dimensi,1,CV_64FC1);
        tmp.ptr<double>(0)[0] = (double)img_data[(i*kol+j)*knl+0];
        tmp.ptr<double>(1)[0] = (double)img_data[(i*kol+j)*knl+1];
        tmp.ptr<double>(2)[0] = (double)img_data[(i*kol+j)*knl+2];
        data_flat.push_back(tmp);
    }
}
cout<<"Data Size:"<<data_flat.size()<<endl;
cout<<"Training..."<<endl;
Mat kluster[3] = {Mat::zeros(img.size(),CV_8UC3),Mat::zeros(img.size(),CV_8UC3),Mat::zeros(img.size(),CV_8UC3)};
uchar* kluster_data[3] = {kluster[0].data,kluster[1].data,kluster[2].data};
jml_data = (int) data_flat.size();
memw = new double[jml_data*jml_komponen];
initGMM();
EM(data_flat,100);
for(int i=0;i<bar;i++){
    for(int j=0;j<kol;j++){
        Mat tmp(dimensi,1,CV_64FC1);
        tmp.ptr<double>(0)[0] = (double)img_data[(i*kol+j)*knl+0];
        tmp.ptr<double>(1)[0] = (double)img_data[(i*kol+j)*knl+1];
        tmp.ptr<double>(2)[0] = (double)img_data[(i*kol+j)*knl+2];
        int lbl=predict_label(tmp);
        kluster_data[lbl][(i*kol+j)*knl+0] = img_data[(i*kol+j)*knl+0];
        kluster_data[lbl][(i*kol+j)*knl+1] = img_data[(i*kol+j)*knl+1];
        kluster_data[lbl][(i*kol+j)*knl+2] = img_data[(i*kol+j)*knl+2];
    }
}
cout<<"Beress!!"<<endl;
imshow("K1",kluster[0]);
imshow("K2",kluster[1]);
imshow("K3",kluster[2]);
waitKey(0);
*/
