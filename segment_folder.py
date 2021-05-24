import pandas as pd
import numpy as np
from PIL import Image
from scipy import stats
from multiprocessing import Pool
import os

def get_mvn(q):
    mvn = []
    limit = []
    mus = []
    data = pd.read_csv("sign_colors_v1.csv")

    for sign in range(1,12):
        data_sign = data.loc[data.sign == sign,["R","G","B"]]
        sigma = np.cov(data_sign,rowvar=False)
        mu = np.mean(data_sign)
        dist = stats.multivariate_normal(mean=mu, cov=sigma)
        mus.append(mu.values.round())
        mvn.append(dist)
        limit.append(np.quantile(dist.logpdf(data_sign),q))
    return mvn, limit, mus

class ImageSegmenter:
    def __init__(self, q): 
        self.mvn, self.log_limit, self.mu = get_mvn(q)

    def segment_image(self, mat):
        height, width = mat.shape[:2]
        seg_mat = np.zeros_like(mat)
        for m in range(height):
            for n in range(width):
                x = mat[m,n,:]
                for dist, limit, mu in zip(self.mvn, self.log_limit, self.mu):
                    if dist.logpdf(x) > limit:
                        seg_mat[m,n,:] = mu
                        break
        return seg_mat

    def work_parallel(self, directories, workers):
        with Pool(processes = workers) as pool:
            pool.map(self.segment_directory, 
                [dir for dir in directories])

        
    def segment_directory(self, directory):
        for file in os.listdir(directory):
            im = Image.open(f"{directory}/{file}") 
            if file.split(".")[-1] == "jpg":
                im = im.transpose(method=Image.ROTATE_270)
            mat = np.asarray(im)
            seg_mat = self.segment_image(mat)
            outdir = f"./model2_easy_segC10/{directory.split('/')[-1]}/{file}"
            Image.fromarray(seg_mat).save(outdir)
            print(outdir)

    
def main():
    workers = 11
    directory = "./model2_easy"
    directories = []
    for dir in os.listdir(directory):
        directories.append(f"{directory}/{dir}")
    q = 0.10
    img_seg = ImageSegmenter(q)
    img_seg.work_parallel(directories, workers)



if __name__=="__main__":
    main()


    
        


