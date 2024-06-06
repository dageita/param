from setuptools import setup


def main():
    package_base = "et_replay"

    # List the packages and their dir mapping:
    # "install_destination_package_path": "source_dir_path"
    package_dir_map = {
        f"{package_base}": ".",
        f"{package_base}.lib": "lib",
        f"{package_base}.lib.comm": "lib/comm",
        f"{package_base}.tests": "tests",
        f"{package_base}.tools": "tools",
    }

    packages = list(package_dir_map)

    setup(
        name="et_replay",
        python_requires=">=3.8",
        author="Louis Feng",
        author_email="lofe@fb.com",
        url="https://github.com/facebookresearch/param",
        packages=packages,
        package_dir=package_dir_map,
    )


if __name__ == "__main__":
    main()
