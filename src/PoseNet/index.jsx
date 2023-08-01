import * as posenet from '@tensorflow-models/posenet'
import * as React from 'react'
import { isMobile, drawKeypoints, drawSkeleton } from './utils'
import ExerciseAnalyzer from './ExerciseAnalyzer'; // 引入ExerciseAnalyzer類別

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
    loadingText: 'Loading pose detector...',
    repsPerSet: 10, // 每組數量，預設10
    totalSets: 3  // 總組數，預設3
  }

  constructor(props) {
    super(props, PoseNet.defaultProps)
    this.state = { 
      loading: true,
      exerciseStage: 'None',     // 當前動作狀態
      state_sequence: [],     // 動作狀態列表,正確順序為[s2,s3,s2]
      correctCount: 0,   // 正確動作次數
      incorrectCount: 0, // 錯誤動作次數
      ExerciseError: false,
      exerciseType: 'squat',
      currentSet: 1,            // 當前組數
      restTimeRemaining: 0,     // 組間休息剩餘時間 (秒)
      isResting: false,         // 是否正在進行組間休息
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
          // 進行指定運動分析
          this.ExerciseAnalyze(keypoints, minPartConfidence);
          this.startRestCountdown();

        }

      })

      // 畫面重複更新
      this.poseDetectionFrameId = requestAnimationFrame(poseDetectionFrameInner);
    }

    poseDetectionFrameInner()
  }

  switchExerciseType = (exerciseType) => {
    this.setState({ exerciseType });
    const sequence = this.state.state_sequence;
    sequence.length = 0
    this.setState({ exerciseStage: 'None', correctCount: 0, incorrectCount: 0, ExerciseError: false, currentSet: 1 });
  };

  ExerciseAnalyze = (keypoints, minPartConfidence) => {
    const { exerciseType } = this.state;
    const currentStage = this.state.exerciseStage; // 動作狀態
    const sequence = this.state.state_sequence; // 狀態list

    // 動作分析
    switch(exerciseType) {
      case 'squat':
        // 深蹲動作分析
        const rightKneeIndex = 14;
        const rightHipIndex = 12;
        const ankleIndex = 16;
        const rightKnee = keypoints[rightKneeIndex].position;
        const rightHip = keypoints[rightHipIndex].position;
        const ankle = keypoints[ankleIndex].position;
        
        const squatDepthElement = document.getElementById('squatDepth');

        // 判斷當前畫面中是否抓到膝蓋和髖部
        if(keypoints[rightHipIndex].score >= minPartConfidence && keypoints[rightKneeIndex].score >= minPartConfidence){
          // 計算膝髖連線與垂直線夾角
          const kneeAngle = Math.abs((90 - Math.atan2(rightKnee.y - rightHip.y, rightKnee.x - rightHip.x) * (180 / Math.PI)));
          
          // 以膝髖連線與垂直線夾角判斷動作狀態
          // 角度<=32度為狀態s1
          if (kneeAngle <= 32) {
            if (currentStage !== 's1') {
              this.setState({ exerciseStage: 's1' }, () => {
                // 判斷動作狀態list的順序是否正確([s2,s3,s2])
                if (
                  sequence.length === 3 &&
                  sequence[0] === 's2' &&
                  sequence[1] === 's3' &&
                  sequence[2] === 's2'
                ) {
                  // 過程中有動作錯誤則增加不正確次數
                  if (this.state.ExerciseError) {
                    this.setState(prevState => ({
                      incorrectCount: prevState.incorrectCount + 1
                    }));
                  } else {
                    this.setState(prevState => ({
                      correctCount: prevState.correctCount + 1
                    }));
                  }
                  // 清空動作錯誤
                  this.setState({ ExerciseError: false});
                }

                sequence.length = 0;
              });
            }
          } else if (kneeAngle >= 35 && kneeAngle <= 65) {
            // 狀態s2
            if (currentStage !== 's2') {
              this.setState({ exerciseStage: 's2' }, () => {
          
                sequence.push('s2');
              });
            }
          } else if (kneeAngle >= 75) {
            // 狀態s3
            if (currentStage !== 's3') {
              this.setState({ exerciseStage: 's3' }, () => {
          
                sequence.push('s3');
              });
            }
            // 處於狀態s3時下蹲角度超過95度判斷動作錯誤
            if(kneeAngle > 95){
              // 顯示動作錯誤提醒並記錄
              this.setState({ ExerciseError: true });
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
            this.setState({ ExerciseError: true });
            ankleAngleElement.innerText = `膝蓋超過腳趾了`;              
          }
          else{
            ankleAngleElement.innerText = ``; 
          }
        }     

        break
      case 'pushup':
        // 伏地挺身分析

    


        break
      case 'bicep-curl':
        // 二頭彎舉分析
        const rightWristIndex = 10;
        const rightElbowIndex = 8;
        const rightShoulderIndex = 6;

        if (
          keypoints[rightWristIndex].score >= minPartConfidence &&
          keypoints[rightElbowIndex].score >= minPartConfidence &&
          keypoints[rightShoulderIndex].score >= minPartConfidence
        ) {
          // 獲取右手腕、右肘和右肩的位置
          const rightWrist = keypoints[rightWristIndex].position;
          const rightElbow = keypoints[rightElbowIndex].position;
          const rightShoulder = keypoints[rightShoulderIndex].position;
    
          // 計算手臂抬起角度
          const armAngle = Math.abs(
            (Math.atan2(rightWrist.y - rightElbow.y, rightWrist.x - rightElbow.x) -
              Math.atan2(rightShoulder.y - rightElbow.y, rightShoulder.x - rightElbow.x)) *
              (180 / Math.PI)
          );

          if (armAngle >= 140){
            if (currentStage !== 's1') {
              this.setState({ exerciseStage: 's1' }, () => {
                // 判斷動作狀態list的順序是否正確([s2,s3,s2])
                if (
                  sequence.length === 3 &&
                  sequence[0] === 's2' &&
                  sequence[1] === 's3' &&
                  sequence[2] === 's2'
                ) {
                  // 過程中有動作錯誤則增加不正確次數
                  if (this.state.ExerciseError) {
                    this.setState(prevState => ({
                      incorrectCount: prevState.incorrectCount + 1
                    }));
                  } else {
                    this.setState(prevState => ({
                      correctCount: prevState.correctCount + 1
                    }));
                  }
                  // 清空動作錯誤
                  this.setState({ ExerciseError: false});
                }

                sequence.length = 0;
              });
            }
          }
          else if(armAngle > 55 && armAngle <= 130){
            // 狀態s2
            if (currentStage !== 's2') {
              this.setState({ exerciseStage: 's2' }, () => {
          
                sequence.push('s2');
              });
            }
          }
          else if(armAngle < 65){
            // 狀態s3
            if (currentStage !== 's3') {
              this.setState({ exerciseStage: 's3' }, () => {
          
                sequence.push('s3');
              });
            }
          }

          // 控制list長度
          if (sequence.length > 3) {
            sequence.shift(); 
          }


          this.setState({ state_sequence: sequence});

          const shoulderToElbowLineAngle = Math.abs(
            (Math.atan2(rightElbow.y - rightShoulder.y, rightElbow.x - rightShoulder.x) - Math.PI / 2) *
              (180 / Math.PI)
          );
          const ankleAngleElement = document.getElementById('ankleAngle');

          if(shoulderToElbowLineAngle > 40){
            this.setState({ ExerciseError: true}, () => {
              ankleAngleElement.innerText = `注意上臂位置`;
            });
          }
          else{
            ankleAngleElement.innerText = ``;
          }
        }
        break
      default: 
        // 預設不做任何事
        break;
    };
  }
  
  startRestCountdown() {
    const { correctCount, incorrectCount, currentSet, restTimeRemaining, isResting, exerciseType } = this.state;
    const { repsPerSet, totalSets } = this.props;

    // 檢查是否達到組數，並且不在休息狀態中
    if (!isResting && correctCount + incorrectCount >= repsPerSet) {
      console.log("helllo");
      // 開始進入組間休息時間
      this.setState({ isResting: true, restTimeRemaining: 90 });

      // 停止運動分析
      this.switchExerciseType('rest');
      // 將 correctCount 和 incorrectCount 歸零
      this.setState({ correctSquatCount: 0, incorrectSquatCount: 0 });
      // 定時器，每秒更新休息時間
      const restTimer = setInterval(() => {
        this.setState(prevState => ({ restTimeRemaining: prevState.restTimeRemaining - 1 }), () => {
          // 檢查休息時間是否結束
          if (this.state.restTimeRemaining === 0) {
            clearInterval(restTimer); // 停止定時器
            this.setState(prevState => ({ 
              currentSet: prevState.currentSet + 1,
              isResting: false,   // 結束休息狀態
            }), () => {
              // 檢查是否達到總組數
              if (this.state.currentSet <= totalSets) {
                // 繼續進行運動分析
                this.switchExerciseType(exerciseType);
              } else {
                // 停止運動分析
                this.switchExerciseType('rest');
                this.setState({ correctSquatCount: 0, incorrectSquatCount: 0 });
              }
            });
          }
        });
      }, 1000);
    }
  }
  

  // 輸出組件
  render() {
    const { currentSet, isResting, restTimeRemaining, incorrectCount, correctCount, exerciseStage  } = this.state;
    const loading = this.state.loading
      ? <div className="PoseNet__loading">{ this.props.loadingText }</div>
      : ''
    return (
      <div>
        <h4>當前狀態: { exerciseStage }/正確次數: { correctCount }/不正確次數: { incorrectCount }/當前組數: {currentSet}/{isResting && <p>休息時間: {restTimeRemaining}秒</p>}</h4>
        {/* 切換深蹲分析的按鈕 */}
        <button onClick={() => this.switchExerciseType('squat')}>深蹲</button>
        {/* 切換伏地挺身分析的按鈕 */}
        <button onClick={() => this.switchExerciseType('pushup')}>伏地挺身</button>
        {/* 切換二頭彎舉分析的按鈕 */}
        <button onClick={() => this.switchExerciseType('bicep-curl')}>二頭彎舉</button>
        <div className="PoseNet">
          { loading }
          <video playsInline ref={ this.getVideo }></video>
          <canvas ref={ this.getCanvas }></canvas>
        </div>        
      </div>

      
    )
  }
}
