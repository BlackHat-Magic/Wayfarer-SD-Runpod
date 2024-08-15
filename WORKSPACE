load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Add bazel_skylib before rules_docker
http_archive(
    name = "bazel_skylib",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz"],
    strip_prefix = "bazel-skylib-1.0.3",
    sha256 = "73b0141ed87833f4fbf17f2e79db77c4b43b869a236ccecf264b09c421cd9a37",
)

http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "b1e80761a8a8243d03ebca8845e9cc1ba6c82ce7c5179ce2b295cd36f7e394bf",
    urls = ["https://github.com/bazelbuild/rules_docker/releases/download/v0.25.0/rules_docker-v0.25.0.tar.gz"],
)

load("@io_bazel_rules_docker//container:container.bzl", "container_repositories")
container_repositories()

load("@io_bazel_rules_docker//repositories:repositories.bzl", container_deps = "deps")
container_deps()
