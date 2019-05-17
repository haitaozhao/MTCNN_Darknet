#include "mtcnn.h"

void run_image(int argc, char **argv)
{
    // load image
    network *pnet = load_mtcnn_net("PNet");
    network *rnet = load_mtcnn_net("RNet");
    network *onet = load_mtcnn_net("ONet");
    printf("\n\n");

    char* filepath = find_char_arg(argc, argv, "--path", "../images/test.jpg");
    if(0==strcmp(filepath, "../images/test.jpg")){
        fprintf(stderr, "Using default: %s\n", filepath);
    }
    params p = initParams(argc, argv);

    image im = load_image_color(filepath, 0, 0);
    im = rgb_to_bgr(im);
    show_im(im, "image", 10);

    int n = 0;
    detect* dets = calloc(0, sizeof(detect));
    double start = 0;
    double endure = 0;

    start = what_time_is_it_now();
    detect_image(pnet, rnet, onet, im, &n, &dets, p);
    endure = what_time_is_it_now() - start;
    printf("Predicted in %.2f seconds. FPS: %.2f\n", endure, 1 / endure);
    show_detect(im, dets, n, "mtcnn", 0, 1, 1, 1);

    free_image(im);
    free(dets);

    free_network(pnet);
    free_network(rnet);
    free_network(onet);
}
