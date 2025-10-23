echo "In order to run this tool, you need to download FLAME. Before you continue, you must register and agree to license terms at:"
echo -e '\e]8;;https://flame.is.tue.mpg.de\ahttps://flame.is.tue.mpg.de\e]8;;\a'

while true; do
    read -p "I have registered and agreed to the license terms at https://flame.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

mkdir -p assets
wget https://huggingface.co/ZeeoRe/SDGTalk/resolve/main/FLAME_masks.pkl -O ./assets/FLAME_masks.pkl
wget https://huggingface.co/ZeeoRe/SDGTalk/resolve/main/FLAME_with_eye.pt -O ./assets/FLAME_with_eye.pt
wget https://huggingface.co/ZeeoRe/SDGTalk/resolve/main/SDGTalk.pt -O ./assets/SDGTalk.pt
wget https://huggingface.co/ZeeoRe/SDGTalk/resolve/main/canonical.obj -O ./assets/canonical.obj
wget https://huggingface.co/ZeeoRe/SDGTalk/resolve/main/demo.zip -O ./demo.zip
wget https://huggingface.co/ZeeoRe/SDGTalk/resolve/main/assets.zip -O ./utils/SDGTalk_track/assets.zip
unzip demo.zip -d ./
rm -r demo.zip
unzip ./utils/SDGTalk_track/assets.zip -d ./utils/SDGTalk_track/
rm -r ./utils/SDGTalk_track/assets.zip
echo "Download complete!"