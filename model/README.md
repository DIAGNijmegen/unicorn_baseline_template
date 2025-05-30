## Upload your model (Recommended approach)

You can upload your method's model separately from the container image as a tarball (.tar.gz) on Grand Challenge under
 **Your algorithm > Models**. This approach is preferred because it is easier to update model weights without rebuilding your Docker image. Additionally, this approach significantly reduces the size of the algorithm's Docker image.


You can use this command to .tar your model:  `tar -czvf algorithmmodel.tar.gz -C model/ .` 

For example, if `algorithmmodel.tar.gz` follows this structure: 
```
model
└── a_tarball_subdirectory
    └── some_tarball_resource.txt
└── README.md
```

At runtime, the content of the model folder will be available in the container at `/opt/ml/model`.