# -*- mode: python -*-

block_cipher = None


a = Analysis(['/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtools/imrestore.py/'],
             pathex=['/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtools', '/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtools'],
             binaries=None,
             datas=None,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='imrestore',
          debug=False,
          strip=False,
          upx=True,
          console=True , version='/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtools/version')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='imrestore')
