/**              projet One_bee_2017
 * @file comptage_video.c
 * @brief compter les entrées/sorties des abeilles avec opencv
 * @author Yekai TANG
 * @date 05/2017
 * @version 1.0
 * @par Université Savoie Mont Blanc ESET L3
 */


#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<stdio.h>
#include<stdlib.h>
#include<opencv2/videoio/videoio.hpp>
#include<iostream>
#include<string.h>
#include <omp.h>        //nous permet d'utiliser openmp
#include <unistd.h>
#include<time.h>
#include<sys/time.h>
#include<mysql/mysql.h>
#include <wiringPi.h>        //nous permet de contrôler les broches GPIO


using namespace std;
using namespace cv;


#define airemin 400        //aire max d'une ligne qui indique la voie
#define airemax 800        //aire min d'une ligne
#define passage 11        //nombre de voies


typedef struct mem
{
int n;
int val[500];
} mem;


void ch_color(VideoCapture cam,int * h_l,int * h_h, int * s_l, int * s_h, int * v_l,int * v_h);
int ch_color_auto(VideoCapture cam,int * h_l,int * h_h, int * s_l, int * s_h, int * v_l,int * v_h);
void traquage(Mat frame, Mat * masque,int * h_l,int * h_h, int * s_l, int * s_h, int * v_l,int * v_h);
void traitimg(Mat * masque,Mat * result);
int nbobjet(Mat masque,vector<vector<Point> > * contours,vector<Vec4i> * hierarchy);
int barycentre(vector<vector<Point> > contours,vector<Vec4i> hierarchy,int * x,int * y,Mat * frame);
int detection_abeille(int * x,int moyl,int * X,int * Y,int * x0,int * y0,Mat * frame,int n,Mat * maskelem);
int dessi_ligne(int * x,int * y,Mat * frame,int n,int ligne,int colone);
Mat calib(VideoCapture cam, int * x, int *y,int * moy,int * n,int *argc,char ***argv);
int cpt(VideoCapture cam,int * x,int moy,int n,int * entre, int * sortie,Mat back,int seuil);
void itoa(int a,char * chaine);
int color_existe(Mat elem,Mat * copie,int h_l,int h_h,int s_l,int s_h,int v_l,int v_h);
void deplacement(Mat * maskelem, int * x1, int * y1,int * x0,int * y0,int n,Mat * frame,int pre,int * deplace,int * x,int moy);
void analyse_deplacement(int * x1, int * y1,int * x0,int * y0,int n,int * deplace,int moyl,int * sens, int * v,int * entre,int * sortie,int * pollen,int * e,int l1,int l2);
int val_hsv(int * A,int * B,int * C,int * D, int * E, int * F,char * guide);
int val(char * guide);


int binarisation(int * x,int n,VideoCapture cam,int moy);
Mat background(int * x,int n,VideoCapture cam,int moy,int seuil);
void compa_back(Mat * elem,Mat * back,Mat * maskelem,int seuil);
int option(char * A,char * guide);
int horloge(int * seconde,int * min,int * hour,int * day, int * months,int * year,tm * tm);
int ascii_to_integer( char *string );




/** ************************************************************************************
 * @brief la fonction principale qui permet d’initialiser la caméra, la librairie wiringPi.
 * et d'appeler la fonction de calibrage et la fonction de comptage
 * @param argc nombre d’instruction
 * @param argv contenue des instructions
 * @return
 *************************************************************************************/
int main(int argc,char **argv)
{
int moy=0,        //moyenne des coordonnées des lignes sur axe y
x[500],y[500],        //stockage des coordonnées
n=0,                //nombre de voie détecté
entre=0,sortie=0,i=0,mem,d=0,
seuil=0;        //seuil utilisé par la méthode d'Otsu


char r = 'y';        //drapeau indiquant si les cordonnées sont correctes
Mat calibrage,backg;
//initialisation de la caméra
VideoCapture cam("./video/v2.mpeg");
//initialisation wiringPi
wiringPiSetup();
pinMode(5,INPUT);


//calibrage automatique


if(cam.isOpened())
{
printf("connection de la caméra réussie\n maintenant commencer le calibrage automatique...\n");
        while(r=='y')
        {
                r = 'n';
                calibrage=calib(cam,x,y,&moy,&n,&argc,&argv);
                printf("resultat du calibrage:\n il y a: %d passages\n axe y de la ligne qui défini entré/sortie: %d \n", n-1, moy);
        //-----------------rangement de données------------------//
                while(d==0)
                {
                d=1;
                        for(i=0;i<(n-1);i++)
                        {
                                if(x[i]>x[i+1])
                                {
                                        mem=x[i];
                                        x[i]=x[i+1];
                                        x[i+1]=mem;
                                        d=0;
                                }
                                if(x[i]<=0||x[i]>640)
                                {
                                        r = 'y';
                                }
                        }
                }


                for(i=0;i<n;i++)
                {
                        printf("x[%d]: %d\n",i,x[i]);
                }
        //------------------fin de rangement------------------------//


        }
        seuil=binarisation(x,n,cam,moy);                //obtension de la valeur de seuil
        backg=background(x,n,cam,moy,seuil);        //obtension du fond à l'aide du seuil
        cpt(cam,x,moy,n,&entre,&sortie,backg,seuil);


        return 0;
}
else
{
printf("aucun flux video est fourni,vérifier le branchement\n");
return -1;
}
}
/** *******************************************************************************
 * @brief binarisation cette fonction est une application de la méthode d'Otsu.
 * le but est de trouver un seuil optimal pour la binarisation du fond.
 * le principe en bref est de synthétiser le fond actuel en histogramme, puis en essayant toutes les
 * possibilités de seuil, on trouve le seuil qui permet de déduire la variance maximale. On peut  trouver
 * les explication de la méthode d'Otsu sur Wikipédia.
 * cette fonction est appliquée dans la zone des voies (zone de comptage), donc on doit récupérer les coordonnées des lignes
 * @param x les abscisses
 * @param n nombre de voie
 * @param cam le flux vidéo
 * @param moy la moyenne des ordonnées
 * @return retourne la valeur du seuil optimale
********************************************************************************/
int binarisation(int * x,int n,VideoCapture cam,int moy)
{
Mat f,frame,gray,bina,median;
int histogram[256]={0},i=0,j=0,d=0,k=0,seuil0=0;


for(k=0;k<20;k++){
cam >> f;
frame=f(Rect(x[0],moy-30,(x[n-1]-x[0]),60));
cvtColor(frame,gray,CV_BGR2GRAY);
        for(j=0;j<256;j++)
        {
                histogram[j]=0;
        }
//établissement de l'histogramme
for( i = 0;i<frame.rows;i++)
{


        uchar * p = (uchar *) frame.ptr<uchar>(i);
        for(j = 0;j<frame.cols;j++)
        {
                histogram[*p++]++;
        }
}


for(i = 0;i <256;i++)
{
        printf("valeur %d: %d\n",i,histogram[i]);
}


//---------calcul d'Ostu--------------------//
int seuil;
    long sum0 = 0, sum1 = 0; //degré(niveau) de gris de la perspective et du fond
    long cnt0 = 0, cnt1 = 0; //nombre de pixels de la perspective et du fond
    double w0 = 0, w1 = 0; //le rapport d'occupation de l'image de la perspective et du fond
    double u0 = 0, u1 = 0;  //le degré(niveau) moyen de gris de la perspective et du fond
    double variance = 0; //la variance
    double u = 0;//le degré(niveau) moyen de gris de l'image entière
    double maxVariance = 0;//variance maximale
    int size = frame.cols*frame.rows;//taille de l'image


    for(i = 1; i < 256; i++)        //parcourir tous les pixels
    {
        sum0 = 0;
        sum1 = 0;
        cnt0 = 0;
        cnt1 = 0;
        w0 = 0;
        w1 = 0;
        for(j = 0; j < i; j++)
        {
            cnt0 += histogram[j];
            sum0 += j * histogram[j];
        }


        u0 = (double)sum0 /  cnt0;
        w0 = (double)cnt0 / size;


        for(j = i ; j <= 255; j++)
        {
            cnt1 += histogram[j];
            sum1 += j * histogram[j];
        }


        u1 = (double)sum1 / cnt1;
        w1 = 1 - w0; // (double)cnt1 / size;


        u = u0 * w0 + u1 * w1;        //le degré (niveau) moyen de gris de l'image entière
        printf("%d :u = %f\n",i, u);
        //variance =  w0 * pow((u0 - u), 2) + w1 * pow((u1 - u), 2);
        variance =  w0 * w1 *  (u0 - u1) * (u0 - u1);
        if(variance > maxVariance)
        {
            maxVariance = variance;
            seuil = i;
        }
    }


    printf("threshold = %d\n", seuil);


if(d==0)
{
        d=1;
        seuil0 = seuil;
}
else
{
        seuil0 = (int)(seuil*0.2 + seuil0*0.8);
}


threshold(gray,bina,seuil,255,CV_THRESH_BINARY_INV);
medianBlur(bina,median,5);


imshow("src",frame);
imshow("gray",gray);
imshow("otsu",median);
waitKey(90);
}


    printf("threshold final= %d\n", seuil0);
        destroyWindow("src");
        destroyWindow("gray");
        destroyWindow("otsu");
        return seuil0;
}


/** **********************************************************************************************
 * @brief background
 * Cette fonction nous permet de trouver le fond de la zone de comptage (zone des voies) en supprimant la perturbation des abeilles
 * On doit récupérer les coordonnées des voies et le seuil qu'on trouve précédemment pour effectuer cette démarche.
 * Dans cette démarche, on utilise la méthode de Surendra:
 * le principe est d’enregistrer une image comme le fond initial, puis on compare le fond binarisé avec l'image actuelle, pour les différents pixels des images binarisées,
 * on change les pixels correspondant dans le fond en fonction de l'image actuelle (le changement de fond respecte l'équation:Nb0 = (1-a)*Nb0 + Nb1*a, vous trouverez les explications ci-dessous ou dans le programme).
 * ( T: le seuil d'Otsu, a: constante entre 0 et 1 définie par nous)
 * (le seuil T est obtenue avec la méthode d'Otsu, on peut connaître cette méthode sur Wiki)
 * 1. enregistrer la première frame en format gris B0 comme l'arrière plan initial
 * binariser B0, le résultat binarisé est M0
 * 2. prélever la prochaine frame et la convertir en format gris, le résultat gris est F
 * binarisé F, le résultat binarisé est M
 * 3.  N1 est l'état (soit 0 soit 255, car elle est binarisée)  du premier pixel dans M et le comparer avec l'état N0 du premier pixel de M0
 *   si la différence du niveau |N1-N0| < T*a
 *  on change la valeur de gris Nb0 de ce pixel pour B0 en fonction de la valeur de gris Nb1 de F : Nb0 = (1-a)*Nb0 + Nb1*a
 *   sinon on doit garder la valeur Nb0
 * 4.ensuite on répète le processus 3 jusqu'à la fin de l'image
 * 5.on répète les processus de 200 à 400 fois (ou encore plus), pour avoir un arrière plan assez propre
 * @param x les abscisses
 * @param n nombre de voies
 * @param cam flux vidéo
 * @param moy la moyenne des ordonnées
 * @param seuil valeur du seuil de binarisation
 * @return retourne le fond qu'on trouve
 **********************************************************************************************/
Mat background(int * x,int n,VideoCapture cam,int moy,int seuil)
{
Mat frame,f,gray,bina,median,B0,B,M0,M;
int i=0,j=0,k=0,delta=0;
cam >> f;
frame=f(Rect(x[0],moy-30,(x[n-1]-x[0]),60));        //tranche la partie d'image (zone de comptage) qui nous intéresse
cvtColor(frame,gray,CV_BGR2GRAY);


threshold(gray,bina,seuil,255,CV_THRESH_BINARY_INV);                //binariser l'image
medianBlur(bina,median,5);                //filtre médian (pour supprimer les petits points de bruit)
B0 = gray.clone();        //enregistrer l'image gris comme le fond initial
M0 = median.clone();        //enregistrer l'image binarisée & filtrée référent qui sera comparée avec M


for(k=0;k<250;k++)        //on prend 250 frames pour déterminer le fond, on peut augmenter la quantité de frame si besoin
{
        cam >> f;
        frame=f(Rect(x[0],moy-30,(x[n-1]-x[0]),60));        //tranche la partie d'image (zone de comptage) qui nous intéresse
        cvtColor(frame,gray,CV_BGR2GRAY);


        threshold(gray,bina,seuil,255,CV_THRESH_BINARY_INV);        //binarisation
        medianBlur(bina,median,5);        //filtre médian
        B = gray.clone();                //enregistrer l'image grise actuelle
        M = median.clone();        //enregistrer l'image binarisée & filtrée actuelle qui sera comparée avec M


        for( i = 0;i<B.rows;i++)
        {
        //déclaration des pointers qui pointent sur les datas (c’est à dire les pixels) des structures d'image
                uchar * p0 = (uchar *) M0.ptr<uchar>(i);
                uchar * p1 = (uchar *) M.ptr<uchar>(i);
                uchar * gp0 = (uchar *) B0.ptr<uchar>(i);
                uchar * gp1 = (uchar *) B.ptr<uchar>(i);


                for(j = 0;j<B.cols;j++)
                {
                        delta = *p0 - *p1;                //comparer un pixel de M0 & M
                        if(delta<0){delta = -delta;}        //prendre la valeur absolue


                        if(delta < (0.3*seuil))                //ici on a un seuil de décision qui dépend du seuil de binarisation: alpha*seuil ici alpha = 0.3, le but est de décider si on change le fond initial
                        {
                                *gp0 = 0.1*(*gp1)+0.9*(*gp0);        //pour changer le fond initial progressivement, on doit respecter une équation: pixel de fond initial = a*pixel actuel + (1-a)*pixel de fond initial,ici on a a=0.1
                        }
                        p0++,p1++,gp0++,gp1++;        //incrémentation des pointeurs pour parcourir tous les pixels
                }
        }


}


        imshow("gray",B0);        //montrer le résultat de fond
        waitKey(200);
        destroyWindow("gray");
return B0;


}


/** ************************************************************************************************
 * @brief compa_back
 * Cette fonction sert à comparer l'image d'une voie de la ruche avec le fond de cette voie, les pixels différent seront remplis par la couleur blanche (valeur:0),
 * les restes seront remplis par la couleur noir (valeur:255), cette démarche nous donnera une image binarisée qui permet au programme de trouver l'abeille.
 * @param elem image actuelle d'une seule voie.
 * @param back image du fond d'une seule voie.
 * @param maskelem image binarisée qui nous intéresse après cette démarche
 * @param seuil le seuil de binarisation qu'on trouve précédemment (avec la méthode d'Otsu)
 **************************************************************************************************/
void compa_back(Mat * elem,Mat * back,Mat * maskelem,int seuil)
{
int i=0,j=0,delta=0;
Mat gray;
cvtColor(*elem,gray,CV_BGR2GRAY);
 *maskelem = back->clone();
                for( i = 0;i<back->rows;i++)
                {
                //les pointeurs qui pointent sur les datas (valeurs des pixels) d'images
                        uchar * p0 = (uchar *) back->ptr<uchar>(i);
                        uchar * p1 = (uchar *) gray.ptr<uchar>(i);
                        uchar * p2 = (uchar *) maskelem->ptr<uchar>(i);


                        for(j = 0;j<back->cols;j++)
                        {
                                delta = *p0 - *p1;
                                if(delta<0){delta = -delta;}
                        //binarisation
                                if(delta < (0.3*seuil))        //le seuil de décision est alpha*seuil, ici alpha = 0.3
                                {
                                        *p2 = 0;
                                }
                                else
                                {
                                        *p2 = 255;
                                }
                                p0++,p1++,p2++;
                        }
                }
        medianBlur(*maskelem,*maskelem,5);
}


/** ************************************************************************************************
 * @brief cpt
 * Cette fonction est le coeur du fonctionnement de programme, il consiste:
 * 1.la préparation du comptage
 *      initialisation de la base de données;
 *      récupération des variables données par l’utilisateur(les options enregistrées dans le fichier de configuration);
 * 2. la boucle infinie du comptage
 *      répartition du multi-processus
 *      traitement d'image des voies
 *      rafraîchissement de la base de données
 * @param cam: flux vidéo
 * @param x: les abscisses
 * @param moy: la moyenne des ordonnées
 * @param n: nombre de voies
 * @param entre: nombre d'abeilles entrées depuis le déclenchement de programme
 * @param sortie: nombre d'abeilles sorties depuis le déclenchement de programme
 * @param back: le fond de la zone de comptage
 * @param seuil: le seuil trouvé par la méthode d'Otsu
 * @return
 ***********************************************************************************************/
int cpt(VideoCapture cam,int * x,int moy,int n,int * entre, int * sortie,Mat back,int seuil)
{
Mat frame, elem[50],maskelem[50],cannyelem[50],copie[50],backcopie[50];


int i=0,e[100],moyl,moyh,ligne,colone,x1[100],y1[100],x0[100],y0[100],pre=0,deplace[100],l1,l2,pollen=0,duree=5;

int entre0 = *entre, sortie0 = *sortie;
//int h_l1=80,h_h1=200, s_l1=40,s_h1=255,v_l1=0,v_h1=90;
int h_l2=30,h_h2=65, s_l2=194,s_h2=241,v_l2=123,v_h2=189;        //la valeur HSV du pollen (on pense que c'est jaune, mais pas encore tester sur la ruche)


int a=0,sens[100],v[100];
int intotal,outtotal;


char r2='n';        //la valeur de r2 dépend de l'option, on va affecter la nouvelle valeur après
//char option1[]="couleurabeille_manuel,",guide1[]="hsv_abeille,",r1='n';
char nom[25],        //nom des fenêtres
guide2[]="hsv_pollen,",option2[]="couleurpollen_manuel,",option3[]="frequence_prelevement_données,";        //ces termes nous permettent de trouver les valeurs dans le fichier ou on enregistre les options demandées par l’utilisateur


time_t temps;
struct tm * tm;


char str1[100],str2[100],str0[100]="insert into compteur values (1,0,0);";        //ce sont les lignes de commande dans Mysql
int seconde0=0,seconde=0,minuite0=0,min=0,hour=0,day=0,months=0,year=0;        //pour récupérer l'horloge
//initialisation de la base de données
MYSQL base;
MYSQL_RES *result;
MYSQL_ROW row;
mysql_init(&base);


if(!mysql_real_connect(&base,"localhost","root","4568525","comptage_video",0,NULL,0)){        //connexion à la base de données
printf("échec à connecter la base de données:%s\n",mysql_error(&base));
}




mysql_query(&base,"CREATE TABLE IF NOT EXISTS `abeille` (   `idAbeille` int(11) NOT NULL AUTO_INCREMENT,   `dateEnregistrement` datetime NOT NULL,   `nbEntreesAbeille` int(11) NOT NULL,   `nbSortiesAbeille` int(11) NOT NULL,   `NBEFauxBourdon` int(11) NOT NULL,   `NBSFauxBourdon` int(11) NOT NULL,   `NBEAbeillePollen` int(11) NOT NULL,   `compteurEntree` bigint(20) NOT NULL,   `compteurSortie` bigint(20) NOT NULL,   PRIMARY KEY (`idAbeille`) ) ENGINE=InnoDB  DEFAULT CHARSET=latin1;");
if(!mysql_query(&base,"CREATE TABLE `compteur` (   `idCompteur` int(11) NOT NULL,   `inTotal` bigint(20) NOT NULL,   `outTotal` bigint(20) NOT NULL,   PRIMARY KEY (`idCompteur`) ) ENGINE=InnoDB DEFAULT CHARSET=latin1;"))
mysql_query(&base,str0);        //si le tableau n'existe pas, on doit insérer une ligne


mysql_query(&base,"select * from compteur;");
result = mysql_use_result(&base);
do{
row = mysql_fetch_row(result);        //obtention d'une ligne de tableau
if(row!=NULL){
    intotal =ascii_to_integer(row[1]);
    outtotal =ascii_to_integer(row[2]);
    printf("%d %d\n",intotal,outtotal);
}
}while(row != NULL);
printf("%d %d\n",intotal,outtotal);


//val_hsv(&h_l1,&h_h1, &s_l1,&s_h1,&v_l1,&v_h1,guide1);




omp_set_num_threads(4);
#pragma omp parallel for        //répartition multi-processus
for(i = 0;i<100;i++)                //initialisation de tous les tableaux
{


        sens[i]=0;
        v[i]=0;
        e[i]=0;
        deplace[i]=0;
        x1[i]=0;
        y1[i]=0;
        x0[i]=0;
        y0[i]=0;


}


//-------lire les options d'utilisateur pour savoir si on doit choisir la couleur d'abeille à la main-------//
//option(&r1,option1);
option(&r2,option2);
duree=val(option3);        //récupérer la fréquence de rafraîchissement de la base de données tableau "compteur"
/*if(r1=='y')
{
        ch_color(cam,&h_l1,&h_h1,&s_l1,&s_h1, &v_l1,&v_h1);
}*/
if(r2=='y')
{
        ch_color(cam,&h_l2,&h_h2,&s_l2,&s_h2, &v_l2,&v_h2);
}
else
    val_hsv(&h_l2,&h_h2, &s_l2,&s_h2,&v_l2,&v_h2,guide2);//récupérere la valeur HSV du pollen
//r1='n';
r2='n';
//---------------------fin----------------//
namedWindow("passage");


for(i=0;i<(n-1);i++)
{
        itoa(i,nom);
        namedWindow(nom);
        printf("%s\n",nom);
}


cam>>frame;
ligne = frame.rows;
colone = frame.cols;
//on veut que chaque voie ait une largeur de 60 donc moyl = moy - 60/2, idem pour moyh
moyl = moy - 30;
moyh = moy +30;
//déterminer les 2 lignes de décision, ils servent à l'analyse du déplacement d'abeille
l1 = moy -10;
l2 = moy +10;




if(moyl<0)
{
        moyl = 0;
}


if(moyh>ligne)
{
        moyl = ligne;
}


for(i=0;i<(n-1);i++)        //trancher l'image complète en plusieurs voies
{
        backcopie[i]=back(Rect(x[i]-x[0],0,(x[i+1]-x[i]),(moyh-moyl)));
}
time(&temps);
tm = localtime(&temps);
//initialisation des 2 compteurs, ils servent à réaliser la fréquence de rafraîchissement de la base de données (enregistrement régulier)
seconde0 = tm->tm_sec;
minuite0 = tm->tm_sec;


while(a!=1048673)        //boucle principale
{
        cam>>frame;


        omp_set_num_threads(4);
        #pragma omp parallel for
        for(i=0;i<(n-1);i++)
        {
                elem[i]=frame(Rect(x[i],moyl,(x[i+1]-x[i]),(moyh-moyl)));        //tranche l'image actuelle d'une seul voie
                //traquage(elem[i],(maskelem+i),&h_l1,&h_h1,&s_l1,&s_h1, &v_l1,&v_h1);
                compa_back(elem+i,backcopie+i,maskelem+i, seuil);
                traitimg((maskelem+i),(cannyelem+i));        //binarisation & calcul canny
                rectangle(frame,Point(x[i],moyl),Point(x[i+1],moyh),Scalar(0,0,255));        //trace un rectangle indiquant la position d'abeille
        }


        deplacement(maskelem,x1,y1,x0,y0,n,&frame,pre,deplace,x,moyl);


for(i=0;i<(n-1);i++)        //détection du pollen
{
        e[i]=color_existe(elem[i],(copie+i),h_l2,h_h2,s_l2,s_h2,v_l2,v_h2);
}
analyse_deplacement(x1, y1, x0,y0, n,deplace,moyl,sens,v, entre,sortie,&pollen, e,l1,l2);
printf("entre : %d sortie:%d somme: %d\n",*entre,*sortie,*entre - *sortie);
printf("nombre de pollen: %d\n",pollen);


time(&temps);
tm = localtime(&temps);
        //printf("compare %d %d\n",tm->tm_sec,seconde);
//enregistrement régulier du 1er tableau dans la base de données
        if(tm->tm_sec == minuite0){
                horloge(&seconde,&min,&hour,&day,&months,&year,tm);
                snprintf(str1,100," insert into abeille values( null,\"%d-%d-%d %d:%d:%d\",%d,%d,0,0,0,%d,%d);",year,months+1,day,hour,min,seconde,*entre-entre0,*sortie-sortie0,(*entre)+intotal,(*sortie)+outtotal);
                entre0 = *entre, sortie0 = *sortie;
                if(mysql_query(&base,str1))
                    printf("échec pour l'insersion");
                    minuite0 = (seconde + 30)%60;
        }


//enregistrement régulièr du 2ème tableau dans la base de données
        if(tm->tm_sec == seconde0)        //teste si on a dépassé la durée donnée pour le tableau 2
        {
                horloge(&seconde,&min,&hour,&day,&months,&year,tm);
                snprintf(str2,100,"update compteur set inTotal=%d,outTotal=%d where idCompteur = 1;",(*entre)+intotal,(*sortie)+outtotal);
                if(mysql_query(&base,str2))
                    printf("échec pour l'insersion");
                seconde0 = (seconde +duree)%60;
        }
        //tracer les lignes de décision
        line(frame, Point(0,l1),Point(colone,l1),Scalar(255,0,0),2);
        line(frame, Point(0,l2),Point(colone,l2),Scalar(255,0,0),2);
        line(frame, Point(0,moy),Point(colone,moy),Scalar(255,0,0),2);
        imshow("passage",frame);
        for(i=0;i<(n-1);i++)
        {
                itoa(i,nom);
                namedWindow(nom);
                imshow(nom,maskelem[i]);
        }
        pre=1;        //lever le drapeau pour indiquer que c'est n'est pas la première boucle
        //tester le signal de la broche 5, si on l'envoi un état haut, on éteindra la RPI3
        if(digitalRead(5))
              system("sudo shutdown now");
        a = waitKey(90);


}
destroyWindow("passage");
mysql_free_result(result);
mysql_close(&base);
return 0;
}
/** *****************************************************************************************************
 * @brief calib
 * Cette partie est la 2ème partie importante dans ce programme, il permet au programme de trouver les positions des voies, à l'aide des traits tracés sur la ruche, on peut réaliser cette démarche,
 * on distingue simplement la couleur des traits pour trouver leur positions. Donc la valeur HSV des traits est indispensable. Pour augmenter le niveau d'automatisation de notre calibrage,
 *  on trouve une méthode qui nous permet de seulement entrer la valeur H(dans H.S.V,H correspond à la couleur rouge, verte, bleu...)
 * le principe de cette méthode est de parcourir toutes les combinaisons de H.S.V, puis on cherche la valeur optimale qui nous permet de trouver le bon nombre des voies,
 * on peut considérer qu'on transfère l'information du nombre des voies aux valeurs S et V.
 * Si on ne veut pas utiliser cette méthode, cette fonction supporte aussi la méthode manuelle (ajuster la couleur par l'utilisateur), ou alors récupérer la valeur par défaut enregistrée
 * @param cam: flux video
 * @param x: les abscisses
 * @param y: les ordonnées
 * @param moy: la moyenne des ordonnées
 * @param n: le nombre des voies
 * @param argc: nombre des paramètres à la ligne de commande
 * @param argv: contenue des paramètres à la ligne de commande
 * @return
 *******************************************************************************************************/
Mat calib(VideoCapture cam, int * x, int *y,int * moy,int * n,int *argc,char ***argv)
{
int ligne=0, colone=0;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
//char t[] = "Voulez-vous choisir la couleur du trait?";
char guide[]="hsv_trait,",option1[]="couleurtrait_auto,";        //terme qui nous permet de trouver les valeurs dans le fichier d'option
Mat frame,masque,canny;
int h_l=0,h_h=18, s_l=105,s_h=175,v_l=85,v_h=135;        //valeur HSV initiale




char r='n';
option(&r,option1);
printf("%c\n",r);
if(r=='y')
{
        ch_color_auto(cam,&h_l,&h_h,&s_l,&s_h, &v_l,&v_h);        //recherche de la luminosité automatique
}
else
{
        val_hsv(&h_l,&h_h,&s_l,&s_h, &v_l, &v_h,guide);        //récupére la valeur HSV par défaut
}
r='n';
cam>>frame;


colone=frame.cols;
ligne=frame.rows;
printf("size image: %d %d\n",colone,ligne);


namedWindow("result");
namedWindow("video");
/*printf("maintenant on commence à tracer le passage, si vous être satisfait pour le résultat vous pouvez appuyer sur 'a' pour passer en prochaine étape\n");
getchar();*/
if(cam.isOpened())
{
        while(*n!=passage)
        {
                cam>>frame;


                //définir la taille d'image qui nous intéresse
                //frame=frame(Rect(100,100,300,300));
                traquage(frame,&masque,&h_l,&h_h,&s_l,&s_h, &v_l,&v_h);
                traitimg(&masque,&canny);
                *n=nbobjet(masque,&contours,&hierarchy);
                printf("%d objet(s) detecté(s)\n",*n);
                *n=barycentre(contours,hierarchy,x,y,&frame);
                printf("%d objet(s) validée(s)\n",*n);
                if(n>0)
                {
                        //printf("%d %d\n",x[0],y[0]);
                }
                *moy=dessi_ligne(x,y,&frame,*n, ligne,colone);
                /*imshow("video",frame);
                imshow("result",masque);
                waitKey(90); */
                usleep(100000);
        }
printf("calibrage automatique est réussi!\n");
}
else
{


printf("initialisation du camera n'est pas réussie, verifier le branchement");


}
destroyWindow("video");
destroyWindow("result");
return frame;
}
/** *****************************************************************************************************
 * @brief ch_color
 * cette fonction nous permet d'ajuster la couleur de détection, on peut voir le résultat de l'image en cours de changement
 * @param cam
 * @param h_l: valeur h min
 * @param h_h: valeur h max
 * @param s_l: valeur s min
 * @param s_h: valeur s max
 * @param v_l: valeur v min
 * @param v_h: valeur v max
 ******************************************************************************************************/
void ch_color(VideoCapture cam,int * h_l,int * h_h, int * s_l, int * s_h, int * v_l,int * v_h)
{
Mat frame,masque,canny;
int a=0,x[500],y[500],n = 0;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
if(cam.isOpened())
{


namedWindow("result");
namedWindow("video");


createTrackbar("h0","video",h_l,255);
createTrackbar("h1","video",h_h,255);
createTrackbar("s0","video",s_l,255);
createTrackbar("s1","video",s_h,255);
createTrackbar("v0","video",v_l,255);
createTrackbar("v1","video",v_h,255);


        while(a!=1048673)
        {
                cam>>frame;


                //définir la taille d'image qui nous intéresse
                //frame=frame(Rect(100,100,300,300));


                traquage(frame,&masque,h_l,h_h,s_l,s_h, v_l,v_h);
                traitimg(&masque,&canny);
                n=nbobjet(masque,&contours,&hierarchy);
                printf("%d objets detectés\n",n);
                n=barycentre(contours,hierarchy,x,y,&frame);
                printf("%d objets validées\n",n);
                imshow("video",frame);
                imshow("result",masque);
                a=waitKey(150);
        }


destroyWindow("video");
destroyWindow("result");
printf("maintenant commencer le traquage de la couleur que vous avez choisi \n");
}




}
/** *****************************************************************************************************
 * @brief traquage
 * cette partie sert à détecter la couleur qui nous intéresse
 * entrée, la détection est effectuée sur les images en format HSV
 * @param frame: image à traiter
 * @param masque: image binarisée
 * @param h_l: valeur h min
 * @param h_h: valeur h max
 * @param s_l: valeur s min
 * @param s_h: valeur s max
 * @param v_l: valeur v min
 * @param v_h: valeur v max
 ******************************************************************************************************/
void traquage(Mat frame, Mat * masque,int * h_l,int * h_h, int * s_l, int * s_h, int * v_l,int * v_h)
{
Mat hsv;


//transforme l'image en format hsv
                cvtColor(frame, hsv, CV_BGR2HSV);
//tri de la couleur
                inRange(hsv,Scalar(*h_l,*s_l,*v_l),Scalar(*h_h,*s_h,*v_h),*masque);
}




/** *****************************************************************************************************
 * @brief traitimg
 * cette partie sert à traiter l'image binarisée
 * on doit:
 * 1.supprimer le bruit
 * 2.trouver les contours d'objets(calcule canny)
 * @param masque: image binarisée
 * @param canny: image canny
 ******************************************************************************************************/
void traitimg(Mat * masque,Mat * canny)
{
Mat kernel;
//définit le kernel
kernel = getStructuringElement(MORPH_ELLIPSE,Size(7,7));
//suppression du bruit
morphologyEx(*masque,*masque,MORPH_CLOSE,kernel);
//trouver les contours
Canny(*masque,*canny,100,200);
}


/** *****************************************************************************************************
 * @brief nbobjet
 * cette fonction nous permet de trouver les contours des objets
 * @param masque: image binarisée
 * @param contours: structure pour enregistrer les contours (en fait c'est un ensembles de pixel)
 * @param hierarchy: structure pour enregistrer
 * @return nombre de contours trouvés
 ******************************************************************************************************/
int nbobjet(Mat masque,vector<vector<Point> > * contours,vector<Vec4i> * hierarchy)
{
Mat canny;
canny = masque.clone();
findContours(canny,*contours,*hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
return (int)contours->size();
}


/** *****************************************************************************************************
 * @brief barycentre
 * Cette fonction sert à trouver les barycentres des objet à partir des contours trouvés
 * @param contours: les contours qu'on trouve avec la fonction nbobjet
 * @param hierarchy: les hiérarchies qu'on trouve avec la fonction nbobjet
 * @param x: les abscisses qu'on va enregistrer
 * @param y: les ordonnées qu'on va enregistrer
 * @param frame: on fournit l'image actuelle pour qu'on puisse marquer les barycentres dessus.
 * @return retourne le nombre des barycentres trouvés
 ******************************************************************************************************/
int barycentre(vector<vector<Point> > contours,vector<Vec4i> hierarchy,int * x,int * y,Mat * frame)
{
Moments moment;
double m01;
double m10;
double m00;
int n = (int)contours.size();
int i=0,j=0;


        for(i=0;i<n;i++)
        {
                //printf("%d\n",i);
                if(hierarchy[i][2]<0)
                {
                        moment = moments((cv::Mat)contours[i]);
                        m01= moment.m01;
                        m10= moment.m10;
                        m00= moment.m00;


                        if((m00<airemax) && (m00>airemin))//tester si l'objet respecter la bonne taille, sinon on considère il est un effet de buits
                        {
                                x[j] = (int) m10/m00;
                                y[j] = (int) m01/m00;


                                //printf("surface %d :%f \n",i,m00);


                                rectangle(*frame,Point(x[i]-10,y[i]-10),Point(x[i]+10,y[i]+10),Scalar(0,255,0));
                                j++;
                        }
                        else
                        {
                                n--;
                        }
                }
        }
return n;
}


/** *****************************************************************************************************
 * @brief detection_abeille
 * Cette fonction nous permet de trouver plusieurs abeilles dans une seule voie, elle a un principe similaire à la fonction barycentre:
 * int barycentre(vector<vector<Point> > contours,vector<Vec4i> hierarchy,int * x,int * y,Mat * frame)
 * Mais la fonction detection_abeille doit connaître l'abeille qui est en train d'entrer/sortir de la ruche,
 * ça veut dire que si on détecte plusieurs abeilles dans la voie, on doit négliger la nouvelle abeille qui entre après l'ancienne.
 * @param x: les abscisses
 * @param moyl: la valeur d'ordonnées minimum de la zone de comptage
 * @param X: abscisses actuelles
 * @param Y: ordonnées actuelles
 * @param x0: dernières abscisses
 * @param y0: dernières ordonnées
 * @param frame: pointeur de l'image actuelle complète
 * @param n: nombre de voies
 * @param maskelem: le masque d'une seule voie
 * @return
 ******************************************************************************************************/
int detection_abeille(int * x,int moyl,int * X,int * Y,int * x0,int * y0,Mat * frame,int n,Mat * maskelem)
{
Moments moment;
double m01;
double m10;
double m00;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
int i=0,j=0,k=0,l=0,x1[100],y1[100],nbre=0,delta1=0,delta2=0;


for(k=0;k<(n-1);k++)                //répéter pour chaque voie
{
        nbobjet(maskelem[k],&contours,&hierarchy);
        nbre = (int)contours.size();
        for(i=0;i<nbre;i++)        //répéter pour chaque objet
        {
                if(hierarchy[i][2]<0)
                {
                        moment = moments(contours[i]);
                        m01= moment.m01;        //x
                        m10= moment.m10;        //y
                        m00= moment.m00;        //surface


                        if(m00>50&&m00<2000)        //200 est la taille minimum d'abeille
                        {
                                x1[j] = (int) m10/m00;
                                y1[j] = (int) m01/m00;




                                rectangle(*frame,Point(*(x1+i)-10+*(x+k),*(y1+i)-10+moyl),Point(*(x1+i)+10+*(x+k),*(y1+i)+10+moyl),Scalar(0,255,0));                //pour tracer les position sur l'image complète en vert
                                j++;
                        }
                        else
                        {
                                nbre--;
                        }
                }
        }        //fin d'analyse des objets dans une voie


        j=0;
                if(nbre==0)
                {
                                Y[k]=0;
                                X[k]=0;
                }
                else
                {
                                Y[k]=y1[0];
                                X[k]=x1[0];
                                for(l=0;l<nbre;l++)
                                {
                                delta2=(y1[l]-y0[k]);
                                if(delta2<0){delta2 = -delta2;}
                                        if(delta1>delta2)//chercher qui est le plus proche
                                        {
                                                Y[k]=y1[l];
                                                X[k]=x1[l];
                                        }
                                delta1=delta2;
                                }
                }
rectangle(*frame,Point(*(X+k)-10+*(x+k),*(Y+k)-10+moyl),Point(*(X+k)+10+*(x+k),*(Y+k)+10+moyl),Scalar(0,0,255));                //trace la position de l’ancienne abeille en rouge
}
return 0;


}
/** *****************************************************************************************************
 * @brief dessi_ligne
 * Cette fonction sert à tracer les lignes de décision sur l'image complète, c'est une partie non importante
 * @param x
 * @param y
 * @param frame
 * @param n
 * @param ligne
 * @param colone
 * @return
 ******************************************************************************************************/
int dessi_ligne(int * x,int * y,Mat * frame,int n,int ligne,int colone)
{
int i=0,som=0,moy=0;
        for(i=0;i<n;i++)
        {
                line(*frame, Point(x[i],0),Point(x[i],ligne),Scalar(255,0,0),2);
        }
        for(i=0;i<n;i++)
        {
                som=som+y[i];
        }
        if(n>0)
        {
                moy=som/n;
        }
        line(*frame, Point(0,moy),Point(colone,moy),Scalar(255,0,0),2);
        return moy;
}


/** *****************************************************************************************************
 * @brief itoa
 *Cette fonction permet de transformer un entier en chaine de caractère
 * @param a
 * @param chaine
******************************************************************************************************/
void itoa(int a,char * chaine)
{
int x;


x=(a/1000)%10;
chaine[0]=x+'0';
x=(a/100) %10;
chaine[1]=x+'0';
x=(a/10)%10;
chaine[2]=x+'0';
x=(a/1)%10;
chaine[3]=x+'0';


chaine[4]=0;


}


/** *****************************************************************************************************
 * @brief color_existe
 * Cette partie sert à réaliser le comptage du pollen, on n'a pas encore vérifié si ça fonctionne dans la pratique,
 * le principe est de traquer la couleur jaune (on suppose que le pollen est jaune)
 * @param elem: l'image actuelle de chaque voie
 * @param copie: masque pour chaque voie
 * @param h_l: valeur h min
 * @param h_h: valeur h max
 * @param s_l: valeur s min
 * @param s_h: valeur s max
 * @param v_l: valeur v min
 * @param v_h: valeur v max
 * @return
*******************************************************************************************************/
int color_existe(Mat elem,Mat * copie,int h_l,int h_h,int s_l,int s_h,int v_l,int v_h)
{


        int x,y;


        traquage(elem,copie,&h_l,&h_h,&s_l,&s_h,&v_l,&v_h);


        //un seul barycentre
        Moments moment;
        double m01;
        double m10;
        double m00;




                        //printf("%d\n",i);
                                moment = moments((cv::Mat)(*copie));
                                m01= moment.m01;
                                m10= moment.m10;
                                m00= moment.m00;
                                if(m00<300)
                                {
                                        x = (int) m10/m00;
                                        y = (int) m01/m00;
                                        //printf("surface pollen: %f\n",m00);
                                }
                                else
                                {
                                        x = 0;
                                        y = 0;
                                }
                                //printf("%d :%d %d      ",i,x[i],y[i]);


        if(x > 0&&y>0)
        {
                return 1;
        }
        else
        {
                return 0;
        }
}


/** *****************************************************************************************************
 * @brief deplacement
 * Cette fonction nous permet de trouver la différence entre les positions actuelles et les dernières positions. Ici on doit tester une variable 'pre' qui indique si c'est la première fois qu'on entre dans la boucle,
 * car lorsque l’on entre pour la première fois dans la boucle, on ne peut pas avoir la dernière position, dans ce cas là on va simplement mettre la position actuelle dans la dernière position, et on ne modifie pas la différence entre la position actuelle et la dernière position.Si on ne fait pas ça, il y aura les erreurs de segmentation.
 * @param maskelem: les masques des voies
 * @param x1 x1 y1: indiquent les positions actuelles
 * @param y1
 * @param x0 x0 y0: indiquent les dernières positions
 * @param y0
 * @param n: nombre de voies
 * @param frame: pointeur de l'image complète
 * @param pre: drapeau
 * @param deplace: différence entre la position actuelle et la position dernière
 * @param x: les abscisses trouvées dans la fonction de calibrage
 * @param moyl: l'ordonnée minimum de la zone de comptage
 ******************************************************************************************************/
void deplacement(Mat * maskelem, int * x1, int * y1,int * x0,int * y0,int n,Mat * frame,int pre,int * deplace,int * x,int moyl)
{
        int i=0;
        if(pre==1)        //on a besoin de cette variable pour éviter les erreurs de segmentation lors de la première boucle
                 //la valeur de pre par défaut est 0, indiquant que c'est la première fois qu'on entre dans la boucle, dans ce cas là notre calcul est différent
        {
                for(i=0;i<(n-1);i++)
                {
                        *(y0+i)=*(y1+i);        //enregistre les dernières positions
                }
        }
        detection_abeille(x,moyl,x1,y1,x0,y0,frame,n,maskelem);        //acquisition des positions actuelles


        if(pre == 1)
        {
                for(i=0;i<(n-1);i++)
                {
                        *(deplace+i) = *(y1+i)-*(y0+i);                //calculer la différence entre la position actuelle et la dernière position
                }
        }
        else
        {
                for(i=0;i<(n-1);i++)
                {
                        *(y0+i)=*(y1+i);        //Si c'est la première fois qu'on entre dans la boucle, on fait juste la copie de la position actuelle dans la dernière position.
                }
        }






}


/** *****************************************************************************************************
 * @brief analyse_deplacement
 * Cette partie est le coeur de la fonction de comptage, toute la partie est exécutée en multi-processus, on simule une machine d'état pour analyser les déplacemets.
 * Pour comprendre le processus de cette machine d'état, vous pouvez regarder le graphe de transition dans la partie comptage vidéo->la méthode de comptage dans le rapport de l'année 2016/2017
 * @param x1 x1 y1: indiquent les positions actuelles
 * @param y1
 * @param x0 x0 y0: indiquent les dernières positions
 * @param y0
 * @param n: nombre de voies
 * @param deplace: le déplacement que l'on trouve avec la fonction deplacement
 * @param moyl: l'ordonnée minimum de la zone de comptage
 * @param sens: s'il vaut 0: la voie est vide; s'il vaut 1: abeille en train d'entrer ;s'il vaut 2: abeille en train de sortir, cette variable indique l'action de l'abeille
 * @param v: une variable de tolérance, elle sert à connaître le cas où le barycentre a disparu (les erreurs)
 * @param entre: nombre d'abeilles entrées
 * @param sortie: nombre d'abeilles sorties
 * @param pollen: nombre d'abeilles avec pollen entrées
 * @param e: s'il vaut 1: abeille avec pollen dans la voie; s'il vaut 0: abeille sans pollen dans la voie
 * @param l1: ligne de décision
 * @param l2: ligne de décision
 ******************************************************************************************************/
void analyse_deplacement(int * x1, int * y1,int * x0,int * y0,int n,int * deplace,int moyl,int * sens, int * v,int * entre,int * sortie,int * pollen,int * e,int l1,int l2)
{
int i=0;


//définition du mode entrée/sortie
omp_set_num_threads(4);
#pragma omp parallel for
for(i=0;i<(n-1);i++)        //répète pour plusieurs voies
{


if(e[i] == 1)
{
        printf("voie %d pollen détecté\n",i);
}
        //-------------la voie est vide-----------------//
        if(sens[i]==0)
        {


                if(deplace[i]>0&&(y1[i]+moyl)>l1&&(y1[i]+moyl)<l2)        //si le déplacement est positif et la position axe y < la ligne au mileu, on passe en mode d'entrée
                {
                                sens[i] = 1;
                }


                if(deplace[i]<0&&(y1[i]+moyl)<l2&&(y1[i]+moyl)>l1)        //si le déplacement est négatif et la position axe y > la ligne au mileu, on passe en mode de sortie
                {
                        sens[i] = 2;
                }
        }
        //-------mode entrée-------//


        if(sens[i]==1)
        {


                if(x1[i]==0&&y1[i]==0)        //si aucune abeille 2 fois, on passe en mode neutre
                {
                        v[i]++;
                }
                if(v[i]>=2)
                {
                        sens[i]=0;
                        v[i]=0;
                }
                else
                {
                        if(deplace[i]>1&&(y1[i]+moyl)>l2)        //avance, donc on compte et on passe en mode neutre
                        {
                                *entre = *entre + 1 ;
                                printf("enre\n");
                                v[i]=0;
                                if(e[i]==1)
                                {
                                        *pollen=*pollen + 1;
                                        (void) pollen;
                                }
                                sens[i]=0;
                        }
                        if(deplace[i]<-1&&(y1[i]+moyl)<l1)        // recule, donc on passe en mode neutre
                        {
                                sens[i]=0;
                        }
                }
        }
        //----------mode sortie-------------//
        if(sens[i]==2)
        {
                if(x1[i]==0&&y1[i]==0)        //si aucune abeille 2 fois, on passe en mode neutre
                {
                        v[i]++;
                }
                if(v[i]>=2)
                {
                        sens[i]=0;
                        v[i]=0;
                }
                else
                {
                        if(deplace[i]<-1&&(y1[i]+moyl)<l1)        //l'abeille avance, donc on compte et on passe en mode neutre (voie est vide)
                        {
                                *sortie = *sortie + 1 ;
                                printf("sortie\n");
                                sens[i]=0;
                                v[i]=0;
                        }
                        if(deplace[i]>1&&(y1[i]+moyl)>l2)        // l'abeille recule, donc on passe en mode neutre (voie est vide)
                        {
                                sens[i]=0;
                        }
                }
        }
}




}


/** *****************************************************************************************************
 * @brief ch_color_auto
 * Cette fonction nous permet de connaître la luminosité si on connaît déjà le nombre de voies.
 * processus:
 * 1.Parcourir toutes les possibilités de la valeur HSV, on note le nombre de voies qu’on trouve avec chaque valeur de HSV.
 * 2.On regroupe toutes les possibilités avec lesquelles on trouve le bon nombre de voies.
 * 3.On teste chaque possibilité dans le groupe avec 10 frames successives pour vérifier la stabilité.
 * 4.On retourne la possibilité la plus stable.
 * Ici pour savoir le nombre des voies, la fonction va lire une variable globale qui peut être définit par l’utilisateur
 * @param cam: flux vidéo
 * @param h_l: valeur h min
 * @param h_h: valeur h max
 * @param s_l: valeur s min
 * @param s_h: valeur s max
 * @param v_l: valeur v min
 * @param v_h: valeur v max
 * @return
 ******************************************************************************************************/
int ch_color_auto(VideoCapture cam,int * h_l,int * h_h, int * s_l, int * s_h, int * v_l,int * v_h)
{
int i=0,j=0,n=0,k=0,stab=0,d=0;
(void) stab;
mem m;        //Cette structure sert à regrouper les possibilités avec lesquelles on trouve le bon nombre de voies
m.n = 0;
Mat frame,masque,canny;
int x[500],y[500];
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
int h1[500],h2[500],s1[500],s2[500],v1[500],v2[500];
                        h1[0]=*h_l;
                        h2[0]=*h_h;
                        s1[0]=*s_l;
                        s2[0]=*s_h;
                        v1[0]=*v_l;
                        v2[0]=*v_h;


if(cam.isOpened())
{
printf("citer tout les possibilités HSV...\n");
for(i=1;s2[i-1] != 255||v2[i-1] != 255;i=i+1)
{


        h1[i]=h1[i-1];
        h2[i]=h2[i-1];
        s1[i]=s1[i-1];
        s2[i]=s2[i-1];
        v1[i]=v1[i-1]+10;
        v2[i]=v2[i-1]+10;
        if(v2[i]>255)
        {
                v2[i]=255;
        }
        if(v2[i]==255&&s2[i]<=255)
        {
                i++;
                h1[i]=h1[i-1];
                h2[i]=h2[i-1];
                s1[i]=s1[i-1]+10;
                s2[i]=s2[i-1]+10;
                v1[i]=*v_l;
                v2[i]=*v_h;
                if(s2[i]>255)
                {
                        s2[i]=255;
                        v1[i]=v1[i-1];
                        v2[i]=v2[i-1];


                }
        }
}




                        cam>>frame;


//------------------citer tout les possibilité avec lesquelles qu'on peut détecter les traits-----------------//
omp_set_num_threads(4);
#pragma omp parallel for private(contours,hierarchy,masque,canny,x,y,k)
        for(j=0;j<i;j++)
        {


                        k = omp_get_thread_num();
                        //printf("thread:%d\n",k);
                        traquage(frame,&masque,h1+j,h2+j,s1+j,s2+j, v1+j,v2+j);
                        traitimg(&masque,&canny);
                        nbobjet(masque,&contours,&hierarchy);
                        n=barycentre(contours,hierarchy,x,y,&frame);
                        if(n == passage)
                        {
                                m.val[m.n]=j;
                                m.n++;
                        }
        }
printf("résultats:\n");
for(j=0;j<m.n;j++)
{
printf("j: %d \n",m.val[j]);
printf("j %d hl %d hh %d sl %d sh %d vl %d vh %d\n",j,h1[m.val[j]],h2[m.val[j]],s1[m.val[j]],s2[m.val[j]],v1[m.val[j]],v2[m.val[j]]);
}
printf("calculer les stabilités...\n");
//------------calculer la stabilité de chaque possibilité--------------//


                for(i=0;i<j;i++)
                {


                                cam>>frame;
                printf("analyser la possibilité numéro %d...\n",i);
                omp_set_num_threads(4);
                #pragma omp parallel for private(masque,canny,contours,hierarchy) reduction(+:d)
                                for(k=0;k<10;k++)
                                {
                                        if(frame.empty()==1){
                                        printf("cannot access frame\n");
                                        }
                                        else
                                        {
                                                traquage(frame,&masque,h1+m.val[i],h2+m.val[i],s1+m.val[i],s2+m.val[i],v1+m.val[i],v2+m.val[i]);
                                                traitimg(&masque,&canny);
                                                nbobjet(masque,&contours,&hierarchy);
                                                n=barycentre(contours,hierarchy,x,y,&frame);
                                                        if(n == passage)
                                                        {
                                                                d++;
                                                        }
                                                waitKey(50);
                                        }
                                }
                                if(stab<d)
                                {
                                        stab = d;
                                        *h_l=h1[m.val[i]];
                                        *h_h=h2[m.val[i]];
                                        *s_l=s1[m.val[i]];
                                        *s_h=s2[m.val[i]];
                                        *v_l=v1[m.val[i]];
                                        *v_h=v2[m.val[i]];
                                }
                }
                printf("succès, le meilleur choix est:\n");
                printf("hl %d hh %d sl %d sh %d vl %d vh %d\n",*h_l,*h_h,*s_l,*s_h,*v_l,*v_h);


}
return -2;
}
/** *****************************************************************************************************
 * @brief val_hsv
 * Cette fonction sert à récupérer les valeurs de HSV enregistré dans le fichier de configuration
 * @param A h_l
 * @param B h_h
 * @param C s_l
 * @param D s_h
 * @param E v_l
 * @param F v_h
 * @param guide: terme de guide pour trouver la chaîne de caractère correctement
 * @return
 ******************************************************************************************************/
int val_hsv(int * A,int * B,int * C,int * D, int * E, int * F,char * guide)
{
        char a[4],b[4],c[4],d[4],e[4],f[4],*data;
        int lng=0,i=0,j=0,size,drap=1;
        //,b[4],c[4],d[4],e[4],f[4]
        lng=strlen(guide);
        printf("longeur terme:%d\n",lng);
        FILE * fi = fopen("./config/configure.csv","r+");
        //fprintf(f,"%s","789654321456");
        fseek(fi,0L,SEEK_END);
        size = ftell(fi);
        rewind(fi);
        printf("taille du fichier:%d\n",size);
        data = (char *)malloc(size + 1);
        if(data == NULL)
        {
                fclose(fi);
                printf("fichier de configuration est vide!!!\n");
                return -1;
        }
        fread(data, size,1,fi);
        printf("recopie succès");
        data[size]=0;
        printf("fin\n");
        printf("%s\n",data);
        for(i=0;(i<(size-lng-1)&&drap==1);i++)
        {
        printf("%d",i);
                drap=0;
                for(j=0;j<lng;j++)
                {
                        if(guide[j]!=data[i+j])
                        {
                                drap=1;
                        }
                }
        }
        printf("terme trouvé i = %d\n",i);
        for(i=i+lng-1;data[i]==' ';i++){}
        printf("i = %d\n",i);
        for(j=0;data[i+j]!=',';j++)
        {
                a[j] = data[i+j];
        }
        a[j]=0;
        printf("%s\n",a);
        i = i + j + 1;
        for(j=0;data[i+j]!=',';j++)
        {
                b[j] = data[i+j];
        }
        b[j]=0;
        i = i + j + 1;
        printf("succès b\n");
        for(j=0;data[i+j]!=',';j++)
        {
                c[j] = data[i+j];
        }
        c[j]=0;
        printf("succès c\n");
        i = i + j + 1;
        for(j=0;data[i+j]!=',';j++)
        {
                d[j] = data[i+j];
        }


        d[j]=0;
        printf("succès d\n");
        i = i + j + 1;
        for(j=0;data[i+j]!=',';j++)
        {
                e[j] = data[i+j];
        }
        e[j]=0;
        printf("succès e\n");
        i = i + j + 1;
        for(j=0;data[i+j]!=' ';j++)
        {
                f[j] = data[i+j];
        }
        f[j]=0;
        printf("succès f\n");
        *A = ascii_to_integer( a );
        *B = ascii_to_integer( b );
        *C = ascii_to_integer( c );
        *D = ascii_to_integer( d );
        *E = ascii_to_integer( e );
        *F = ascii_to_integer( f );
printf("vous voulez prendre les valeurs par défault\n du coup les valeurs %s par défaut:%s %s %s %s %s %s\n",guide,a,b,c,d,e,f);


        fclose(fi);
return 0;


}


/** *****************************************************************************************************
 * @brief val
 * Cette fonction sert à trouver les options dans le fichier de configuration
 * @param guide: terme de guide pour trouver la chaîne de caractère correctement
 * @return retourne la valeur trouvée
*******************************************************************************************************/
int val(char * guide)
{
char a[4],*data;
        int lng=0,i=0,j=0,size,drap=1,valeur;
        //,b[4],c[4],d[4],e[4],f[4]
        lng=strlen(guide);
        printf("longeur terme:%d\n",lng);
        FILE * fi = fopen("./config/configure.csv","r+");
        //fprintf(f,"%s","789654321456");
        fseek(fi,0L,SEEK_END);
        size = ftell(fi);
        rewind(fi);
        printf("taille du fichier:%d\n",size);
        data = (char *)malloc(size + 1);
        if(data == NULL)
        {
                fclose(fi);
                printf("fichier de configuration est vide!!!\n");
                return -1;
        }
        fread(data, size,1,fi);
        printf("recopie succès");
        data[size]=0;
        printf("fin\n");
        printf("%s\n",data);
        for(i=0;(i<(size-lng-1)&&drap==1);i++)
        {
        printf("%d",i);
                drap=0;
                for(j=0;j<lng;j++)
                {
                        if(guide[j]!=data[i+j])
                        {
                                drap=1;
                        }
                }
        }
        printf("terme trouvé i = %d\n",i);
        for(i=i+lng-1;data[i]==' ';i++){}
        printf("i = %d\n",i);
        for(j=0;data[i+j]!=' ';j++)
        {
                a[j] = data[i+j];
        }
        a[j]=0;
        printf("%s\n",a);


        printf("succès f\n");
        valeur = ascii_to_integer( a );
        fclose(fi);
return valeur;




}


/** *****************************************************************************************************
 * @brief option
 * Cette fonction sert à trouver les options dans le fichier de configuration
 * @param A la valeur trouvée
 * @param guideterme de guide pour trouver la chaîne de caractère correctement
 * @return
 ******************************************************************************************************/
int option(char * A,char * guide)
{
        char a[4],*data;
        int lng=0,i=0,j=0,size,drap=1,valeur;
        //,b[4],c[4],d[4],e[4],f[4]
        lng=strlen(guide);
        printf("longeur terme:%d\n",lng);
        FILE * fi = fopen("./config/configure.csv","r+");
        //fprintf(f,"%s","789654321456");
        fseek(fi,0L,SEEK_END);
        size = ftell(fi);
        rewind(fi);
        printf("taille du fichier:%d\n",size);
        data = (char *)malloc(size + 1);
        if(data == NULL)
        {
                fclose(fi);
                printf("fichier de configuration est vide!!!\n");
                return -1;
        }
        fread(data, size,1,fi);
        printf("recopie succès");
        data[size]=0;
        printf("fin\n");
        printf("%s\n",data);
        for(i=0;(i<(size-lng-1)&&drap==1);i++)
        {
        printf("%d",i);
                drap=0;
                for(j=0;j<lng;j++)
                {
                        if(guide[j]!=data[i+j])
                        {
                                drap=1;
                        }
                }
        }
        printf("terme trouvé i = %d\n",i);
        for(i=i+lng-1;data[i]==' ';i++){}
        printf("i = %d\n",i);
        for(j=0;data[i+j]!=' ';j++)
        {
                a[j] = data[i+j];
        }
        a[j]=0;
        printf("%s\n",a);


        printf("succès f\n");
        valeur = ascii_to_integer( a );
        if(valeur == 1)
        {
                *A = 'y';
        }
        else
        {
                *A = 'n';
        }
        fclose(fi);
return 0;


}


/** *****************************************************************************************************
 * @brief horloge
 * Cette fonction sert à affecter les valeurs de date dans chaque variable
 * @param seconde
 * @param min
 * @param hour
 * @param day
 * @param months
 * @param year
 * @param tm: structure qui enregistre la date
 * @return
 ******************************************************************************************************/
int horloge(int * seconde,int * min,int * hour,int * day, int * months,int * year,tm * tm)
{
*seconde = tm->tm_sec;
*min = tm->tm_min;
*hour = tm->tm_hour;
*day = tm->tm_mday;
*months = tm->tm_mon;
*year = tm->tm_year + 1900;
return 1;
}
//----------------conversion des données-------------//
/** *****************************************************************************************************
 * @brief ascii_to_integer
 * Cette fonction sert à convertir une chaîne de caractère en entier
 * @param string
 * @return
*******************************************************************************************************/
int ascii_to_integer( char *string )
{
    int value; value = 0;
    while( *string >= '0' && *string <= '9' ){
        value *= 10;
        value += *string - '0';
        string++;
    }


    if( *string != '\0' )
        {
                value = 0;
        }
    return value;
}
