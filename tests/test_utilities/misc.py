from pybinding.tests.utils.fuzzy_equal import FuzzyEqual
from pybinding.tests.utils.path import path_from_fixture


def temp_file_from_tmp_path(request, tmp_path, ext=''):
    """Use a fixture's `request` argument to create a unique file name in a temporary system folder defined with
    `tmp_path`

    The final path looks like:
        tmp_path/test_name[fixture_param]variant.ext"""
    name = request.node.name.replace('test_', '')

    file_name = (tmp_path / name).with_suffix(ext)
    return file_name

