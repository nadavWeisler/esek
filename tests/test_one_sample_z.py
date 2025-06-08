from src import utility
from src import Calculator


def test_always_true():
    assert True


def test_convert_results(data):

    result = Calculator.OneSampleMean.one_sample_from_parameters(**data)
    converted_result = utility.convert_results(result)
    return f"{converted_result=}"


if __name__ == "__main__":
    data = {
        "sample_mean": 140,
        "sample_size": 30,
        "population_mean": 100,
        "population_sd": 15,
    }
    print(test_convert_results(data))
