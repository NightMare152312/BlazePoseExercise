import * as posenet from '@tensorflow-models/posenet'
import * as React from 'react'
import { isMobile, drawKeypoints, drawSkeleton } from './utils'

export default class PoseNet extends React.Component {

  static defaultProps = {
    videoWidth: 600,
    videoHeight: 500,
    flipHorizontal: true,
    algorithm: 'single-pose',
    mobileNetArchitecture: isMobile() ? 0.50 : 1.01,
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
    minPoseConfidence: 0.2,
    minPartConfidence: 0.6,
    maxPoseDetections: 2,
    nmsRadius: 20.0,
    outputStride: 16,
    imageScaleFactor: 0.5,
    skeletonColor: 'aqua',
    skeletonLineWidth: 2,
    loadingText: 'Loading pose detector...'
  }

  constructor(props) {
    super(props, PoseNet.defaultProps)
    this.state = { 
      loading: true,
      squatStage: 'None',     // 當前動作狀態
      state_sequence: [],     // 動作狀態列表,正確順序為[s2,s3,s2]
      correctSquatCount: 0,   // 正確動作次數
      incorrectSquatCount: 0, // 錯誤動作次數
      kneeToToeError: false,  // 動作錯誤:膝蓋超過腳趾
      squatDepthError: false, // 動作錯誤:蹲姿過低
    }
  }

  getCanvas = elem => {
    this.canvas = elem
  }

  getVideo = elem => {
    this.video = elem
  }

  async componentWillMount() {
    // Loads the pre-trained PoseNet model
    this.net = await posenet.load(this.props.mobileNetArchitecture);
  }

  async componentDidMount() {
    try {
      await this.setupCamera()
    } catch(e) {
      throw 'This browser does not support video capture, or this device does not have a camera'
    } finally {
      this.setState({ loading: false })
    }

    this.detectPose()
  }

  async setupCamera() {
      // MDN: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw 'Browser API navigator.mediaDevices.getUserMedia not available'
    }

    const { videoWidth, videoHeight } = this.props
    const video = this.video
    const mobile = isMobile()

    video.width = videoWidth
    video.height = videoHeight

    // MDN: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: 'user',
        width: mobile ? void 0 : videoWidth,
        height: mobile ? void 0: videoHeight,
      }
    });

    video.srcObject = stream

    return new Promise(resolve => {
      video.onloadedmetadata = () => {
        // Once the video metadata is ready, we can start streaming video
        video.play()
        resolve(video)
      }
    })
  }

  detectPose() {
    const { videoWidth, videoHeight } = this.props
    const canvas = this.canvas
    const ctx = canvas.getContext('2d')

    canvas.width = videoWidth
    canvas.height = videoHeight

    this.poseDetectionFrame(ctx)
  }

  poseDetectionFrame(ctx) {
    const {
      algorithm,
      imageScaleFactor,
      flipHorizontal,
      outputStride,
      minPoseConfidence,
      maxPoseDetections,
      minPartConfidence,
      nmsRadius,
      videoWidth,
      videoHeight,
      showVideo,
      showPoints,
      showSkeleton,
      skeletonColor,
      skeletonLineWidth,
    } = this.props

    const net = this.net
    const video = this.video

    const poseDetectionFrameInner = async () => {
      let poses = [];

      // 單人姿勢估計or多人姿勢估計(採用單人)
      switch (algorithm) {
        case 'single-pose':

          const pose = await net.estimateSinglePose(
            video,
            imageScaleFactor,
            flipHorizontal,
            outputStride
          )

          poses.push(pose)

          break
        case 'multi-pose':

          poses = await net.estimateMultiplePoses(
            video,
            imageScaleFactor,
            flipHorizontal,
            outputStride,
            maxPoseDetections,
            minPartConfidence,
            nmsRadius
          )

          break
      }

      ctx.clearRect(0, 0, videoWidth, videoHeight);

      if (showVideo) {
        ctx.save()
        ctx.scale(-1, 1)
        ctx.translate(-videoWidth, 0)
        ctx.drawImage(video, 0, 0, videoWidth, videoHeight)
        ctx.restore()
      }

      // For each pose (i.e. person) detected in an image, loop through the poses
      // and draw the resulting skeleton and keypoints if over certain confidence
      // scores
      poses.forEach(({ score, keypoints }) => {
        if (score >= minPoseConfidence) {
          // 畫出各部位的姿勢估計
          if (showPoints) {
            drawKeypoints(keypoints, minPartConfidence, skeletonColor, ctx);
          }
          if (showSkeleton) {
            drawSkeleton(keypoints, minPartConfidence, skeletonColor, skeletonLineWidth, ctx);
          }

          const currentStage = this.state.squatStage; // 動作狀態
          const sequence = this.state.state_sequence; // 狀態list
          const rightKneeIndex = 14;
          const rightHipIndex = 12;
          const ankleIndex = 16;
          const rightKnee = keypoints[rightKneeIndex].position;
          const rightHip = keypoints[rightHipIndex].position;
          const ankle = keypoints[ankleIndex].position;
          
          const squatDepthElement = document.getElementById('squatDepth');

          // 判斷當前畫面中是否抓到膝蓋和髖部
          if(keypoints[rightHipIndex].score >= minPartConfidence && keypoints[rightKneeIndex].score >= minPartConfidence){
            // 計算膝髖連線與垂直線夾腳
            const kneeAngle = Math.abs((90 - Math.atan2(rightKnee.y - rightHip.y, rightKnee.x - rightHip.x) * (180 / Math.PI)));
            
            // 以膝髖連線與垂直線夾角判斷動作狀態
            // 角度<=32度為狀態s1
            if (kneeAngle <= 32) {
              if (currentStage !== 's1') {
                this.setState({ squatStage: 's1' }, () => {
                  console.log('目前狀態：s1');
                  // 判斷動作狀態list的順序是否正確([s2,s3,s2])
                  if (
                    sequence.length === 3 &&
                    sequence[0] === 's2' &&
                    sequence[1] === 's3' &&
                    sequence[2] === 's2'
                  ) {
                    // 過程中有動作錯誤兩者其一則增加不正確次數
                    if (this.state.kneeToToeError || this.state.squatDepthError) {
                      this.setState(prevState => ({
                        incorrectSquatCount: prevState.incorrectSquatCount + 1
                      }));
                    } else {
                      this.setState(prevState => ({
                        correctSquatCount: prevState.correctSquatCount + 1
                      }));
                    }
                    // 清空動作錯誤
                    this.setState({ squatDepthError: false, kneeToToeError: false});
                  }

                  sequence.length = 0;
                });
              }
            } else if (kneeAngle >= 35 && kneeAngle <= 65) {
              // 狀態s2
              if (currentStage !== 's2') {
                this.setState({ squatStage: 's2' }, () => {
                  console.log('目前狀態：s2');
            
                  sequence.push('s2');
                });
              }
            } else if (kneeAngle >= 75) {
              // 狀態s3
              if (currentStage !== 's3') {
                this.setState({ squatStage: 's3' }, () => {
                  console.log('目前狀態：s3');
            
                  sequence.push('s3');
                });
              }
              // 處於狀態s3時下蹲角度超過95度判斷動作錯誤
              if(kneeAngle > 95){
                // 顯示動作錯誤提醒並記錄
                this.setState({ squatDepthError: true });
                squatDepthElement.innerText = `蹲得太下去了`;   
              }
              else{
                squatDepthElement.innerText = ``;
              }
            }
            
            // 控制list長度
            if (sequence.length > 3) {
              sequence.shift(); 
            }


            this.setState({ state_sequence: sequence});
          }

          // 判斷動作過程中膝蓋是否超過腳趾
          if(keypoints[ankleIndex].score >= minPartConfidence){
            const ankleAngle = Math.atan2(ankle.y - rightKnee.y, ankle.x - rightKnee.x) * (180 / Math.PI) - 90;
            const ankleAngleElement = document.getElementById('ankleAngle');
            //const kneeangleElement = document.getElementById('kneeAngle');
            // Test
            //kneeangleElement.innerText = `Ankle to Knee Angle: ${ankleAngle.toFixed(2)}°`;

            if(Math.abs(ankleAngle) > 30){
              // 腳踝膝蓋連線與垂直線夾角超過30度時判斷動作錯誤
              this.setState({ kneeToToeError: true });
              ankleAngleElement.innerText = `膝蓋超過腳趾了`;              
            }
            else{
              ankleAngleElement.innerText = ``;  
            }


          }

        }
      })

      // 畫面重複更新
      requestAnimationFrame(poseDetectionFrameInner)
    }

    poseDetectionFrameInner()
  }

  // 輸出組件
  render() {
    const squatStage = this.state.squatStage;
    const correctSquatCount = this.state.correctSquatCount;
    const incorrectSquatCount = this.state.incorrectSquatCount;
    const loading = this.state.loading
      ? <div className="PoseNet__loading">{ this.props.loadingText }</div>
      : ''
    return (
      <div>
        <h2>當前狀態: { squatStage } / 正確次數: { correctSquatCount } / 不正確次數: { incorrectSquatCount }</h2>
        <div className="PoseNet">
          { loading }
          <video playsInline ref={ this.getVideo }></video>
          <canvas ref={ this.getCanvas }></canvas>
        </div>        
      </div>

      
    )
  }
}
