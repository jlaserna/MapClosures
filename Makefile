.PHONY: cpp

ifeq (,$(wildcard .env))
    $(error .env file not found)
endif

include .env
export $(shell sed 's/=.*//' .env)


install:
	@pip install --verbose ./python/

uninstall:
	@pip -v uninstall map-closures

editable:
	@pip install scikit-build-core pyproject_metadata pathspec pybind11 ninja cmake
	@pip install --no-build-isolation -ve ./python/

cpp:
	@cmake -Bbuild cpp/ -DGITHUB_TOKEN=$(GITHUB_TOKEN)
	@cmake --build build -j$(nproc --all)

ext:
	@cmake -Bbuild python/ -DGITHUB_TOKEN=$(GITHUB_TOKEN)
	@cmake --build build -j$(nproc --all)

clean:
	@rm -rf build/ dist/ *.egg-info
