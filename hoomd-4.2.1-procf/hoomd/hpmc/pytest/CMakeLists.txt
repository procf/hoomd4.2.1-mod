# copy python modules to the build directory to make it a working python package
set(files __init__.py
          test_boxmc_move_tuner.py
          test_clusters.py
          test_compute_free_volume.py
          test_compute_sdf.py
          test_external_field.py
          test_external_user.py
          test_external_wall.py
          test_muvt.py
          test_boxmc.py
          test_shape.py
          test_shape_updater.py
          test_shape_utils.py
          test_move_size_tuner.py
          test_pair_user.py
          test_pair_union_user.py
          test_quick_compress.py
          test_scale.py
          test_small_box_2d.py
          test_small_box_3d.py
          conftest.py
          depletants_sphere.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/hpmc/pytest
       )

copy_files_to_build("${files}" "hpmc_pytest" "*.py")
