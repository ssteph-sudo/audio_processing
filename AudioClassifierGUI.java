import javax.sound.sampled.*;
import javax.swing.*;
import java.awt.event.*;
import java.io.File;
import java.io.IOException;

public class AudioClassifierGUI {

    private JFrame frame;
    private JList<String> fileList;
    private JButton btnSelectFolder;
    private JButton btnPlayAudio;
    private JScrollPane scrollPane;
    private DefaultListModel<String> listModel;
    private JLabel lblStatus;

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                AudioClassifierGUI window = new AudioClassifierGUI();
                window.frame.setVisible(true);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    public AudioClassifierGUI() {
        initialize();
    }

    private void initialize() {
        frame = new JFrame();
        frame.setBounds(100, 100, 500, 350);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().setLayout(null);

        btnSelectFolder = new JButton("Select Folder");
        btnSelectFolder.addActionListener(e -> {
            JFileChooser folderChooser = new JFileChooser();
            folderChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            if (folderChooser.showOpenDialog(frame) == JFileChooser.APPROVE_OPTION) {
                File selectedFolder = folderChooser.getSelectedFile();
                displayAudioFiles(selectedFolder);
            }
        });
        btnSelectFolder.setBounds(10, 11, 150, 23);
        frame.getContentPane().add(btnSelectFolder);

        listModel = new DefaultListModel<>();
        fileList = new JList<>(listModel);
        scrollPane = new JScrollPane(fileList);
        scrollPane.setBounds(10, 45, 464, 205);
        frame.getContentPane().add(scrollPane);

        btnPlayAudio = new JButton("Play Selected Audio");
        btnPlayAudio.addActionListener(e -> playSelectedAudio());
        btnPlayAudio.setBounds(10, 261, 150, 23);
        frame.getContentPane().add(btnPlayAudio);

        lblStatus = new JLabel("Status: Idle");
        lblStatus.setBounds(10, 290, 300, 25);
        frame.getContentPane().add(lblStatus);
    }

    private void displayAudioFiles(File folder) {
        listModel.clear(); // Clear previous entries
        addAudioFiles(folder);
    }

    private void addAudioFiles(File folder) {
        File[] files = folder.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    addAudioFiles(file); // Recursively search in subdirectories
                } else if (file.getName().toLowerCase().endsWith(".wav")) {
                    listModel.addElement(file.getAbsolutePath());
                }
            }
        }
    }

    private void playSelectedAudio() {
        String selectedFile = fileList.getSelectedValue();
        if (selectedFile != null) {
            lblStatus.setText("Status: Playing " + selectedFile); // Update status
            new Thread(() -> {
                try {
                    playWAV(selectedFile);
                    lblStatus.setText("Status: Finished Playing"); // Update status after playback
                } catch (Exception ex) {
                    lblStatus.setText("Status: Error"); // Update status on error
                    ex.printStackTrace();
                }
            }).start();
        } else {
            lblStatus.setText("Status: No File Selected"); // Update status if no file selected
        }
    }

    private void playWAV(String filePath) {
        try {
            File audioFile = new File(filePath);
            AudioInputStream audioStream = AudioSystem.getAudioInputStream(audioFile);
            Clip clip = AudioSystem.getClip();
            clip.open(audioStream);
            clip.start();
    
            // Wait for the playback to complete
            while (!clip.isRunning())
                Thread.sleep(10);
            while (clip.isRunning())
                Thread.sleep(10);
    
            clip.close();
        } catch (UnsupportedAudioFileException | IOException | LineUnavailableException | InterruptedException e) {
            e.printStackTrace();
            lblStatus.setText("Status: Error Playing File");
        }
    }
}
