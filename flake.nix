{
  description = "SOS PoC A — CPU semantic filesystem search";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        python = pkgs.python312;

        # Everything the PoC needs at runtime.
        pyDeps = ps: [
          ps.numpy
          ps.pillow
          ps.watchdog
          ps.python-magic
          (ps.torchWithoutCuda or ps.torch)
          ps.open-clip-torch
        ];

        # Tests that use a fake encoder need no model and no network.
        pyTestDeps = ps: [ ps.numpy ps.pillow ps.watchdog ];

        pyEnv = python.withPackages pyDeps;
        pyTestEnv = python.withPackages pyTestDeps;

        sqliteVec = pkgs.sqlite-vec or null;
        haveVec = sqliteVec != null;
        vecPath = pkgs.lib.optionalString haveVec "${sqliteVec}/lib";

        mkApp = name: script: {
          type = "app";
          program = pkgs.lib.getExe (pkgs.writeShellApplication {
            inherit name;
            runtimeInputs = [ pyEnv pkgs.tesseract ];
            text = ''
              export SQLITE_VEC_PATH="${vecPath}"
              export PYTHONPATH="${./.}:''${PYTHONPATH:-}"
              exec python "${script}" "$@"
            '';
          });
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ pyEnv pkgs.sqlite pkgs.file pkgs.tesseract ]
            ++ pkgs.lib.optional haveVec sqliteVec;

          shellHook = ''
            export PYTHONPATH="$PWD:''${PYTHONPATH:-}"
            export PATH="$PWD/bin:$PATH"
            export SQLITE_VEC_PATH="${vecPath}"
            echo "SOS PoC A — $(python --version)"
            if [ -n "$SQLITE_VEC_PATH" ]; then
              echo "  vector backend : sqlite-vec ($SQLITE_VEC_PATH)"
            else
              echo "  vector backend : numpy brute force"
            fi
            echo "  ocr            : $(tesseract --version 2>&1 | head -1)"
            echo "  index          : sos-index <dir> --db sos.db"
            echo "  search         : sos-search \"a cat\" --timings    (--repl for many)"
            echo "  score          : python eval/score.py --db sos.db"
            echo "  tests          : python -m unittest discover -s tests -v"
          '';
        };

        # Fast shell for the fake-encoder tests: no torch, no model download.
        devShells.test = pkgs.mkShell {
          packages = [ pyTestEnv pkgs.sqlite pkgs.tesseract ]
            ++ pkgs.lib.optional haveVec sqliteVec;
          shellHook = ''
            export PYTHONPATH="$PWD:''${PYTHONPATH:-}"
            export SQLITE_VEC_PATH="${vecPath}"
          '';
        };

        # `nix flake check` runs the suite in the sandbox, both vector backends.
        checks.tests = pkgs.runCommand "sospoc-tests"
          {
            nativeBuildInputs = [ pyTestEnv pkgs.tesseract ]
              ++ pkgs.lib.optional haveVec sqliteVec;
          }
          ''
            cp -r ${./.} src && chmod -R u+w src && cd src
            export PYTHONPATH="$PWD"
            export HOME=$TMPDIR

            echo "--- numpy brute-force backend ---"
            SOS_FORCE_NUMPY=1 python -m unittest discover -s tests -v

            echo "--- sqlite-vec backend ---"
            SQLITE_VEC_PATH="${vecPath}" python -m unittest discover -s tests -v

            touch $out
          '';

        apps = {
          index = mkApp "sos-index" ./bin/sos-index;
          search = mkApp "sos-search" ./bin/sos-search;
          score = mkApp "sos-score" ./eval/score.py;
        };
      });
}
