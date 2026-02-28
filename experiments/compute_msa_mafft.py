from pathlib import Path
import subprocess
import sys


def main():
    # Project root
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    input_fasta = BASE_DIR / "data" / "curated-AMPs.fasta"
    output_dir = BASE_DIR / "results" / "msa"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_fasta = output_dir / "aligned_mafft.fasta"

    if not input_fasta.exists():
        print(f"Input file not found: {input_fasta}")
        sys.exit(1)

    print("Running MAFFT alignment...")
    print(f"Input:  {input_fasta}")
    print(f"Output: {output_fasta}")

    try:
        with open(output_fasta, "w") as outfile:
            subprocess.run(
                ["mafft", "--auto", str(input_fasta)],
                stdout=outfile,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
    except subprocess.CalledProcessError as e:
        print("MAFFT failed.")
        print(e.stderr)
        sys.exit(1)

    print("Alignment completed successfully.")


if __name__ == "__main__":
    main()