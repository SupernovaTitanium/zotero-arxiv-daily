"""Tests for zotero_arxiv_daily.utils: glob_match, normalization."""

from zotero_arxiv_daily.utils import glob_match


# ---------------------------------------------------------------------------
# glob_match — migrated from test_glob_match.py
# ---------------------------------------------------------------------------


class TestGlobMatch:
    """Test cases for the glob_match function."""

    def test_exact_match(self):
        assert glob_match("hello.txt", "hello.txt")
        assert not glob_match("hello.txt", "world.txt")
        assert glob_match("", "")

    def test_wildcard_asterisk(self):
        assert glob_match("hello.txt", "*.txt")
        assert not glob_match("hello.py", "*.txt")
        assert glob_match("file", "*")

        assert glob_match("hello.world.txt", "*.*.txt")
        assert not glob_match("hello.txt", "*.*.txt")
        assert glob_match("a.b.c.d", "*.*.*.*")

        assert glob_match("hello_world.txt", "hello*world.txt")
        assert glob_match("hello123world.txt", "hello*world.txt")
        assert glob_match("helloworld.txt", "hello*world.txt")
        assert not glob_match("hello_universe.txt", "hello*world.txt")

    def test_wildcard_question_mark(self):
        assert glob_match("hello.txt", "hell?.txt")
        assert not glob_match("hell.txt", "hell?.txt")
        assert glob_match("hello.txt", "he??o.txt")
        assert glob_match("heXXo.txt", "he??o.txt")
        assert not glob_match("heo.txt", "he??o.txt")

    def test_character_classes(self):
        assert glob_match("file1.txt", "file[123].txt")
        assert glob_match("file2.txt", "file[123].txt")
        assert not glob_match("file4.txt", "file[123].txt")

        assert glob_match("file1.txt", "file[1-3].txt")
        assert glob_match("file2.txt", "file[1-3].txt")
        assert not glob_match("file4.txt", "file[1-3].txt")

        assert glob_match("fileA.txt", "file[!123].txt")
        assert not glob_match("file1.txt", "file[!123].txt")

    def test_path_separators(self):
        assert glob_match("dir/file.txt", "dir/file.txt")
        assert glob_match("dir/file.txt", "*/file.txt")
        assert glob_match("dir/subdir/file.txt", "*/subdir/file.txt")
        assert glob_match("dir/subdir/file.txt", "dir/*/file.txt")
        assert glob_match("dir/subdir/file.txt", "*/*/file.txt")

    def test_complex_patterns(self):
        assert glob_match("test1_file.txt", "test[1-3]*file.txt")
        assert glob_match("test2_long_file.txt", "test[1-3]*file.txt")
        assert not glob_match("test4_file.txt", "test[1-3]*file.txt")

        assert glob_match("prefix_middle_suffix.log", "prefix*middle*.log")
        assert not glob_match("prefix_other_suffix.log", "prefix*middle*.log")

    def test_edge_cases(self):
        assert glob_match("", "**")
        assert not glob_match("", "?")
        assert not glob_match("file", "")

        assert glob_match("file-name.txt", "file-name.txt")
        assert glob_match("file_name.txt", "file_name.txt")
        assert glob_match("file.name.txt", "file.name.txt")

        assert not glob_match("File.txt", "file.txt")
        assert not glob_match("FILE.TXT", "file.txt")
        assert glob_match("file.txt", "file.txt")

    def test_partial_matches(self):
        assert glob_match("hello.txt", "hello.txt")
        assert not glob_match("prefix_hello.txt", "hello.txt")
        assert glob_match("hello.txt", "*.txt")

    def test_special_glob_characters(self):
        assert not glob_match("file[1].txt", "file[1].txt")
        assert glob_match("file1.txt", "file[1].txt")

    def test_numeric_patterns(self):
        assert glob_match("file001.txt", "file???.txt")
        assert not glob_match("file01.txt", "file???.txt")
        assert glob_match("version1.2.3.txt", "version*.txt")
        assert glob_match("version1.2.3.txt", "version?.?.?.txt")

    def test_extension_patterns(self):
        assert glob_match("document.pdf", "*.pdf")
        assert glob_match("image.jpg", "*.jpg")
        assert glob_match("script.py", "*.py")
        assert not glob_match("data.csv", "*.txt")

        assert glob_match("file.txt", "*.[tc][xs][tv]")
        assert glob_match("file.csv", "*.[tc][xs][tv]")
        assert not glob_match("file.pdf", "*.[tc][xs][tv]")

    def test_recursive_wildcard(self):
        assert glob_match("file.txt", "**/*.txt")
        assert glob_match("dir/file.txt", "**/*.txt")
        assert glob_match("dir/subdir/file.txt", "**/*.txt")
        assert glob_match("dir/subdir/subsubdir/file.txt", "**/*.txt")


# ---------------------------------------------------------------------------
# normalize_title / normalize_doi — dedup key normalization
# ---------------------------------------------------------------------------


def test_normalize_title_strips_case_and_punctuation():
    from zotero_arxiv_daily.utils import normalize_title
    assert normalize_title("A  Self-Attention: Revisited!") == "aselfattentionrevisited"
    assert normalize_title("Hello World") == "helloworld"
    assert normalize_title(None) == ""
    assert normalize_title("") == ""


def test_normalize_doi_strips_url_prefix_and_case():
    from zotero_arxiv_daily.utils import normalize_doi
    assert normalize_doi("https://doi.org/10.1101/2026.03.01.1") == "10.1101/2026.03.01.1"
    assert normalize_doi("DOI:10.26434/ABC") == "10.26434/abc"
    assert normalize_doi("10.1/x") == "10.1/x"
    assert normalize_doi(None) == ""
