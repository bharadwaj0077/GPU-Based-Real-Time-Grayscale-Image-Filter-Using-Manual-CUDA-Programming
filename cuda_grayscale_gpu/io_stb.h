#pragma once
#include "stb_image.h"
#include "stb_image_write.h"

unsigned char* load_image(const char* filename, int& w, int& h, int& c);
void save_gray(const char* filename, unsigned char* data, int w, int h);
void free_image(unsigned char* data);
