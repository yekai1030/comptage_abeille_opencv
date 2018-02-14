
LIBS=-lopencv_core -lopencv_imgcodecs -lopencv_videoio -lopencv_highgui -lopencv_imgproc `pkg-config --cflags --libs gtk+-2.0` -fopenmp -lwiringPi `mysql_config --cflags --libs`

comptage_video: copie.cc
	g++ $^ -o $@ $(LIBS)
clean:
	rm -f *.o
	rm -f comptage_video
