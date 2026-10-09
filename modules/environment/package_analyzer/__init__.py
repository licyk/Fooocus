"""Python 软件包分析工具包

提供 Python 软件包的版本比较、依赖解析、安装验证等功能.

模块分层结构:
    底层:
        - ``py_whl_parse``: PEP 508 解析器基础设施
        - ``py_ver_cmp``: PEP 440 版本比较器
        - ``ver_cmp``: 通用版本比较器
    中层:
        - ``version_utils``: 版本字符串工具 (canonical 检查、包名/版本提取)
        - ``wheel_parser``: Wheel 文件名解析
        - ``requirement_parser``: PEP 508 依赖声明解析与 marker 评估
    高层:
        - ``requirement_normalizer``: 依赖声明标准化 (组合中层模块完成 requirements 列表标准化)
    最高层:
        - ``installation_checker``: 安装状态检查与依赖验证
"""

# 版本字符串工具
# 依赖分类
from modules.environment.package_analyzer.dependency_categorizer import (
    PackageDependencies,
    format_requirement,
    get_categorized_dependencies,
)

# 安装状态检查
from modules.environment.package_analyzer.installation_checker import (
    check_version_constraint,
    get_missing_package_metadata_dependencies,
    get_package_version_from_library,
    is_package_installed,
    parse_package_spec,
    validate_package_metadata_dependencies,
    validate_requirements,
)

# PEP 440 版本比较
from modules.environment.package_analyzer.py_ver_cmp import (
    PyWhlVersionComparison,
    PyWhlVersionComponent,
    PyWhlVersionMatcher,
)

# Requirements 解析
from modules.environment.package_analyzer.py_whl_parse import (
    ParsedPyWhlRequirement,
    RequirementParser,
    get_parse_bindings,
)
from modules.environment.package_analyzer.requirement_normalizer import (
    parse_requirement_list,
)
from modules.environment.package_analyzer.requirement_parser import (
    evaluate_marker,
    parse_requirement,
    parse_requirement_to_list,
    read_packages_from_requirements_file,
)

# 通用版本比较
from modules.environment.package_analyzer.ver_cmp import (
    CommonVersionComparison,
    version_decrement,
    version_increment,
)
from modules.environment.package_analyzer.version_utils import (
    get_correct_package_name,
    get_package_name,
    get_package_version,
    get_package_version_specs,
    is_package_has_version,
    is_prerelease_version,
    normalize_package_name,
    parse_version_component,
    remove_optional_dependence_from_package,
    version_string_is_canonical,
)

# Wheel 文件名解析
from modules.environment.package_analyzer.wheel_parser import (
    parse_wheel_filename,
    parse_wheel_to_package_name,
    parse_wheel_version,
)

__all__ = [
    # ver_cmp
    "CommonVersionComparison",
    # dependency_categorizer
    "PackageDependencies",
    # requirement_parser / py_whl_parse
    "ParsedPyWhlRequirement",
    # py_ver_cmp
    "PyWhlVersionComparison",
    "PyWhlVersionComponent",
    "PyWhlVersionMatcher",
    "RequirementParser",
    "check_version_constraint",
    "evaluate_marker",
    "format_requirement",
    "get_categorized_dependencies",
    "get_correct_package_name",
    "get_missing_package_metadata_dependencies",
    "get_package_name",
    "get_package_version",
    # installation_checker
    "get_package_version_from_library",
    "get_package_version_specs",
    "get_parse_bindings",
    "is_package_has_version",
    "is_package_installed",
    "is_prerelease_version",
    "normalize_package_name",
    "parse_package_spec",
    "parse_requirement",
    "parse_requirement_list",
    "parse_requirement_to_list",
    "parse_version_component",
    # wheel_parser
    "parse_wheel_filename",
    "parse_wheel_to_package_name",
    "parse_wheel_version",
    "read_packages_from_requirements_file",
    "remove_optional_dependence_from_package",
    "validate_package_metadata_dependencies",
    "validate_requirements",
    "version_decrement",
    "version_increment",
    # version_utils
    "version_string_is_canonical",
]
