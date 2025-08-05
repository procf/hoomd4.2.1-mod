# Installing Base HOOMD-blue v4.2.1 (no modifications)

If you need to install the base version of HOOMD-blue 4.2.1 without modification, you can follow these steps:


Download the basic version of HOOMD-blue as a tarball. Tarballs are available as GitHub release assets, as described on the [HOOMD-blue docs](https://hoomd-blue.readthedocs.io/en/stable/building.html#obtain-the-source). On the page for the version you want to install, copy the Download link and use it to curl the tar file.
```bash
curl -Lo hoomd-v4.2.1.tar.gz [link-to-tar, ex:https://github.com/glotzerlab/hoomd-blue/releases/download/v4.2.1/hoomd-v4.2.1.tar.gz]
tar -xvf hoomd-v4.2.1.tar.gz
rm hoomd-v4.2.1.tar.gz
```
If you are backing up to a new GitHub repository you may also need to remove the sphinx-doc folder
```bash
rm -r hoomd-v4.2.1/sphinx-doc
```

Optionally, you can add back the sphinx-doc folder from this repository
```bash
cp -r [path-to-hoomd4.2.1-mod]/hoomd-v4.2.1-procf/sphinx-doc /hoomd-v4.2.1-procf/sphinx-doc
```
<br>

