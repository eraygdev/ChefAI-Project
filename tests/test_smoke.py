"""
ChefAI - Temel smoke test.

Amac: CI'da pytest'in bos test seti ile hata vermesini engellemek.
Model asamasi ilerledikce gercek testler bu klasore eklenecek.
"""


def test_smoke() -> None:
    """Proje iskeleti saglikli mi?"""
    assert True


def test_src_importable() -> None:
    """src paketi import edilebilmeli."""
    import src

    assert src is not None
