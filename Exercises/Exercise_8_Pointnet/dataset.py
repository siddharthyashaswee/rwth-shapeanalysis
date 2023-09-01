import os
import ssl
import zipfile
import shutil
import random
import open3d as o3d
import numpy as np
import torch
from six.moves import urllib

def normalize_unit_sphere(points):
    mean = np.mean(points, axis=-1)
    out = points - mean[:,None]
    radius = np.linalg.norm(out, axis=-2).max(axis=-1)
    out = out / (radius * 2)
    return out

class ModelNet10(torch.utils.data.Dataset):
    
    def download_dataset(self):
        url = 'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
        filename = url.rpartition('/')[2].split('?')[0]
        path = os.path.join(self.path, filename)        
        os.makedirs(self.path)

        context = ssl._create_unverified_context()
        data = urllib.request.urlopen(url, context=context)

        with open(path, 'wb') as f:
            f.write(data.read())
            with zipfile.ZipFile(path, 'r') as f:
                f.extractall(self.path)
         
        os.unlink(path)
        
        metadata_folder = os.path.join(self.path, '__MACOSX')
        if os.path.exists(metadata_folder):
            shutil.rmtree(metadata_folder)
                
        folder = os.path.join(self.path, 'ModelNet10')
        file_names = os.listdir(folder)    
        for file_name in file_names:
            shutil.move(os.path.join(folder, file_name), self.path)
        shutil.rmtree(folder)
    
    def sample_meshes(self, n_points):
        subfolders = [f.path for f in os.scandir(self.path) if f.is_dir()]
        for folder in subfolders: 
            splits = [f.path for f in os.scandir(folder) if f.is_dir()]
            for i in range(3):
                if i == 0:
                    split = folder + '/train'
                    pc_path = folder + "/train_pc.npy"
                elif i == 1:
                    split = folder + '/train'
                    pc_path = folder + "/val_pc.npy"
                elif i == 2:
                    split = folder + '/test'
                    pc_path = folder + "/test_pc.npy"
                files = [f for f in os.listdir(split) if os.path.isfile(os.path.join(split, f)) and f.endswith('.off')]
                if i == 0:
                    perm = np.random.permutation(len(files))# the validation set should be a random subset
                    #files = files[perm]                    
                    files = [files[i] for i in perm]
                    files = files[:int(0.9*len(files))]
                elif i == 1:
                    #files = files[perm]    
                    files = [files[i] for i in perm]
                    files = files[int(0.9*len(files)):]
                    
                point_clouds = []
                for file in files:
                    mesh = o3d.io.read_triangle_mesh(split + "/" + file)
                    pc = mesh.sample_points_uniformly(number_of_points=n_points)
                    points = np.asarray(pc.points)
                    point_clouds.append(normalize_unit_sphere(points.transpose(1,0)))
                point_clouds = np.stack(point_clouds)
                with open(pc_path, 'wb') as f:
                    np.save(f, point_clouds)
            
        
    
    def __init__(self, path, mode):
        super(ModelNet10, self).__init__()
        
        self.path = path
        
        if not os.path.exists(self.path):
            print("Download ModelNet10")
            self.download_dataset()
            print("Sample Meshes")
            self.sample_meshes(1024)
            print("done")
        
        subfolders = [f.path for f in os.scandir(self.path) if f.is_dir()]
        
        self.point_clouds = []
        self.class_ids = []
        self.class_names = []
        for i, folder in enumerate(subfolders): 
            self.class_names.append(folder.split('/')[-1])
            pcs = np.load(folder + "/" + mode + "_pc.npy")
            self.point_clouds.append(torch.from_numpy(pcs).float())
            self.class_ids.append(torch.ones(pcs.shape[0], dtype=torch.long) * i)
        self.point_clouds = torch.cat(self.point_clouds)
        self.class_ids = torch.cat(self.class_ids)
        
    def __getitem__(self, idx):
        return self.point_clouds[idx], self.class_ids[idx]
    
    def __len__(self):
        return self.point_clouds.shape[0]
            
