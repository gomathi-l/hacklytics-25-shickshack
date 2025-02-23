# backend/__init__.py

def main():
    from .hybrid import main as hybrid_main
    hybrid_main()

if __name__ == "__main__":
    main()
