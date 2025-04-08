const clientId = crypto.randomUUID();
const video = document.getElementById('video');
const audio = document.getElementById('audio');
const configuration = {
    sdpSemantics: 'unified-plan'
};
let timerInterval;

let pc; // RTCPeerConnection
let ws; // WebSocket
let savedOffer = null; // Lưu Offer có đầy đủ ICE candidates lần đầu tiên

// function parseIceCandidate(candidate) {
//     const parts = candidate.split(' ');
    
//     const ip = parts[4];
//     const port = parts[5];
//     const protocol = parts[2];
//     const priority = parts[3];
//     const foundation = parts[0].split(':')[1];
//     const component = parts[1] === "1" ? "rtp" : "rtcp";
//     const type = parts[7];

//     return new RTCIceCandidate({
//         ip: ip,
//         port: port,
//         protocol: protocol,
//         priority: priority,
//         foundation: foundation,
//         component: component,
//         type: type,
//         sdpMid: "0",
//         sdpMLineIndex: "0"
//     });
// }

function negotiate() {
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });

    // if (!savedOffer) {
    if (true) {
        console.log("Generating new Offer...");
        return pc.createOffer().then((offer) => {
            return pc.setLocalDescription(offer);
        }).then(() => {
            // wait for ICE gathering to complete
            return new Promise((resolve) => {
                if (pc.iceGatheringState === 'complete') {
                    resolve();
                } else {
                    const checkState = () => {
                        if (pc.iceGatheringState === 'complete') {
                            pc.removeEventListener('icegatheringstatechange', checkState);
                            resolve();
                        }
                    };
                    pc.addEventListener('icegatheringstatechange', checkState);
                }
            });
        }).then(() => {
            savedOffer = pc.localDescription; // Lưu Offer có đầy đủ ICE candidates lần đầu tiên
        //     return fetch('/offer', {
        //         body: JSON.stringify({
        //             sdp: offer.sdp,
        //             type: offer.type,
        //         }),
        //         headers: {
        //             'Content-Type': 'application/json'
        //         },
        //         method: 'POST'
        //     });
        // }).then((response) => {
        //     return response.json();
        // }).then((answer) => {
        //     return pc.setRemoteDescription(answer);

            // let text = document.getElementById('text-input').value;
            console.log("Sending offer to server");
            ws.send(JSON.stringify({
                type: "offer",
                sdp: savedOffer.sdp,
                // text: text
            }));
            document.getElementById("generate").disabled = false;
        }).catch((e) => {
            alert("Error in negotiation: " + e);
        });
    } else {
        // console.log("Using saved offer");
        // return pc.createOffer().then((offer) => {
        //     return pc.setLocalDescription(savedOffer);
        // }).then(() => {
        //     let text = document.getElementById('text-input').value;
        //     ws.send(JSON.stringify({
        //         type: "offer",
        //         sdp: savedOffer.sdp,
        //         text: text
        //     }));
        // }).catch((e) => {
        //     alert("Error in negotiation: " + e);
        // });
    }
}

// Hàm khởi tạo RTCPeerConnection và WebSocket
function initializeConnection() {
    // Đảm bảo đóng kết nối cũ trước khi bắt đầu kết nối mới
    stopStream();

    if (document.getElementById('use-stun').checked) {
        configuration.iceServers = [
            {
              urls: [
                  'stun:stun.l.google.com:19302',
                //   'stun:stun1.l.google.com:19302',
                //   'stun:stun2.l.google.com:19302',
              ],
            },
          ];
          configuration.iceCandidatePoolSize = 20;
    }
    pc = new RTCPeerConnection(configuration);
    // ws = new WebSocket(`ws://localhost:8000/ws/${clientId}`);
    ws = new WebSocket('wss://cicada-eager-hound.ngrok-free.app/ws/${clientId}'); // use: ngrok http --url=cicada-eager-hound.ngrok-free.app 8000

    // Xử lý khi có ICE candidate được tạo
    pc.onicecandidate = (event) => {
        if (event.candidate) {
            console.log("Gathered ICE candidate:", event.candidate);
            // console.log("Sending ICE candidate to server");
            // ws.send(JSON.stringify({
            //     type: 'candidate',
            //     candidate: event.candidate
            // }));
        }
    };

    // Xử lý khi connection state thay đổi
    pc.onconnectionstatechange = function () {
        console.log("Connection state change:", pc.connectionState);
        if (pc.connectionState === "connected") {
            console.log("Peers successfully connected");
        } else if (pc.connectionState === "failed") {
            console.error("Connection failed");
        }
    };

    // Xử lý khi ICE connection state thay đổi
    pc.oniceconnectionstatechange = function () {
        console.log("ICE connection state:", pc.iceConnectionState);
        if (pc.iceConnectionState === "connected") {
            console.log("ICE connection successful");
        } else if (pc.iceConnectionState === "failed") {
            console.error("ICE connection failed");
        }
    };
    
    // Xử lý khi signaling state thay đổi
    pc.onsignalingstatechange = function () {
        console.log("Signaling state change:", pc.signalingState);
    };

    // Xử lý khi ICE gathering state thay đổi
    pc.onicegatheringstatechange = function () {
        console.log("ICE gathering state changed to:", pc.iceGatheringState);
    };

    // Xử lý khi nhận được track video từ server
    pc.ontrack = function (event) {
        if (event.track.kind === 'video') {
            console.log("Video track received");
            video.srcObject = event.streams[0];
            // video.play();
        }
        else {
            console.log("Audio track received");
            audio.srcObject = event.streams[0];
        }

        clearInterval(timerInterval);
        let button = document.getElementById("generate");
        button.classList.remove("loading");
        button.innerHTML = "Generate Avatar";
        button.disabled = false;
        timerDisplay = document.getElementById("timer");
        timerDisplay.style.color = "#333";
    };

    // Đợi WebSocket mở
    ws.onopen = function () {
        console.log("WebSocket connection established");
        let text = document.getElementById('text-input').value;
        ws.send(JSON.stringify({
            type: "text",
            text: text
        }));
        negotiate();
    };

    // Xử lý tin nhắn từ WebSocket
    ws.onmessage = async function (event) {
        const message = JSON.parse(event.data);
        if (message.type === 'answer') {
            console.log("Received answer, setting remote description");
            await pc.setRemoteDescription(new RTCSessionDescription(message));
        } else if (message.type === 'offer') {
            // if (pc.signalingState === 'closed') {
            //     console.error("RTCPeerConnection is closed, cannot set remote description");
            //     return;
            // }
            // Đặt Offer từ server
            console.log("Received offer from server");
            await pc.setRemoteDescription(new RTCSessionDescription(message));
            // Tạo Answer
            const answer = await pc.createAnswer();
            await pc.setLocalDescription(answer);
            // Gửi Answer về server
            console.log("Sending answer to server");
            ws.send(JSON.stringify({
                type: "answer",
                sdp: answer.sdp
            }));
        } else if (message.type === 'candidate') {
            // if (pc.signalingState === 'closed') {
            //     console.error("RTCPeerConnection is closed, cannot add ICE candidate");
            //     return;
            // }
            console.log("Received ICE candidate from server");
            // await pc.addIceCandidate(parseIceCandidate(message.candidate));
            // try {
            //     await pc.addIceCandidate(new RTCIceCandidate(message.candidate));
            // } catch (e) {
            //     console.error("Error adding received ICE candidate", e);
            // }
        } else if (message.type === 'error') {
            console.error("Error from server:", message.error);
        }
    };

    ws.onerror = function (error) {
        console.error("WebSocket error:", error);
    };

    ws.onclose = function () {
        console.log("WebSocket connection closed");
    };
}

// // Hàm bắt đầu stream video
// async function startStream(source) {
//     // Khởi tạo kết nối mới
//     initializeConnection();
// }

// Hàm dừng stream video
function stopStream() {
    if (ws) {
        ws.close(); // Đóng WebSocket
    }
    if (pc) {
        pc.close(); // Đóng RTCPeerConnection
    }
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop()); // Dừng các track video
        video.srcObject = null; // Xóa video stream
    }
    if (audio.srcObject) {
        audio.srcObject.getTracks().forEach(track => track.stop()); // Dừng các track audio
        audio.srcObject = null; // Xóa audio stream
    }
    console.log("Stream stopped");
}

// Biến để quản lý MediaRecorder
let mediaRecorder;
let recordedChunks = [];
// const recordingOptions = {
//     audioBitsPerSecond: 128000,
//     videoBitsPerSecond: 2500000,
//     mimeType: "video/mp4",
// };
const recordingOptions = {
    audioBitsPerSecond: 192000,  // Tăng bitrate audio cho âm thanh rõ hơn
    videoBitsPerSecond: 5000000, // Tăng bitrate video (5Mbps) để giảm nhòe
    mimeType: "video/webm; codecs=vp9", // WebM với VP9 giúp nén tốt hơn, giảm lag
};

// Hàm bắt đầu ghi lại video
function startRecording(stream) {
    if (!stream) {
        console.error("No stream available to record");
        return;
    }
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream, recordingOptions);
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };
    mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, recordingOptions);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'recorded-video.mp4';
        a.click();
        URL.revokeObjectURL(url);
    };
    mediaRecorder.start();
    console.log("Recording started");
}

// Hàm dừng ghi lại video
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        console.log("Recording stopped");
    }
}

function generateAvatar() {
    // Khởi tạo kết nối mới
    initializeConnection();
}

// document.getElementById('start-camera').onclick = () => startStream('camera');
// document.getElementById('start-file').onclick = () => startStream('file');
// document.getElementById('stop-stream').onclick = () => stopStream();
document.getElementById('start-recording').onclick = () => startRecording(video.srcObject);
document.getElementById('stop-recording').onclick = () => stopRecording();
document.getElementById("generate").addEventListener("click", function () {
    let button = this;
    let timerDisplay = document.getElementById("timer");
    let counter = 0;

    button.classList.add("loading");
    button.disabled = true;
    if (!button.querySelector(".loading-text")) {
        button.innerHTML = `<span class="loading-text">Processing...</span>`;
    }

    timerDisplay.textContent = `⏳ 0s`; // Reset đồng hồ
    timerDisplay.style.color = "#007bff"; // Đổi màu để báo hiệu đang chạy
    timerInterval = setInterval(() => {
        counter++;
        timerDisplay.textContent = `⏳ ${counter}s`;
    }, 1000);
    
    generateAvatar();
});
// Điều chỉnh chiều cao textarea khi nhập text
document.getElementById('text-input').addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 250) + 'px'; // Giới hạn chiều cao tối đa
});

