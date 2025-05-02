## Upload your model (Recommneded approach)

You can upload your method's model separately from the container image as a tarball (.tar.gz) on Grand Challenge under
 **Your algorithm > Models**. This approach is preferred because it's easier to update model weights without rebuilding your Docker image.


You can use this command to .tar your model:  `tar -czvf algorithmmodel.tar.gz a_tarball_subdirectory` 

For example, if `algorithmmodel.tar.gz` follows this structure: 
```
a_tarball_subdirectory/
└── some_tarball_resource.txt
```

At runtime, it be available in the container as:

`/opt/ml/model/a_tarball_subdirectory/some_tarball_resource.txt `


