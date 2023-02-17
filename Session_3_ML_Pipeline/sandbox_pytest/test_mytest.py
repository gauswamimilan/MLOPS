from mytest import square
import pytest

# a way to create a parameter value as a function argument
@pytest.fixture
def my_fixture():
    return 2


def test_square():
    assert square(2) == 4
    assert square(3) == 9


def test_square_wih_fixure(my_fixture):
    assert square(my_fixture) == 4
    assert square(3) == 9
